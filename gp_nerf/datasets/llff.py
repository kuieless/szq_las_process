import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from nerf.utils import get_rays
from pathlib import Path
from tools.segment_anything import sam_model_registry, SamPredictor


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose



def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses

def init(device):
    sam_checkpoint = "tools/segment_anything/sam_vit_h_4b8939.pth"
    
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__()
        
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.dataset_path
        self.preload = True # opt.preload # preload data into GPU
        self.scale = 0.33  # opt.scale # camera radius scale to make sure camera are inside the bounding box. default = 0.33
        self.offset = [0, 0, 0]  # opt.offset # camera offset
        self.bound = 1 # opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = True  # opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = opt.sample_ray_num

        self.rand_pose = -1  # opt.rand_pose

        self.enable_semantic = opt.enable_semantic

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            self.f_paths = []
            self.depths = []
            self._sam_features = []
            self.imgs = []

            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)
                self.f_paths.append(f_path)

                if self.enable_semantic:
                    img = cv2.imread(f_path)
                    self.imgs.append(img)

                    depth_path = Path(f_path).parent.parent / 'depths' / f"{Path(f_path).stem}.npy"
                    depth = np.load(depth_path)  # zyq   depth_path
                    depth = torch.HalfTensor(depth)
                    self.depths.append(depth)
                    
                    sam_feature_path = Path(f_path).parent.parent / 'sam_features' / f"{Path(f_path).stem}.npy"
                    feature = np.load(sam_feature_path)
                    feature = torch.from_numpy(feature)
                    self._sam_features.append(feature)

        if self.enable_semantic:
            self.predictor = init(self.device)
            self.select_origin = True
            self.num_depth_process = 0
            self.object_id=0
            self.K1 = None
            self.world_point = None
            self.N_total = opt.sample_ray_num
            self.visualization=True # False  True
            self.num_save = 0

            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        # self.nerf = nerf


    def __len__(self) -> int:
        return self.poses.shape[0]





    def __getitem__(self, index):
        occluded_threshold=0.01
        # print(index)
        poses1 = self.poses[index].to(self.device).unsqueeze(0) # [B, 4, 4]
        
        if self.type == 'train' and not self.enable_semantic:
            rays, directions = get_rays(poses1, self.intrinsics, self.H, self.W, self.num_rays)
        else:
            rays, directions = get_rays(poses1, self.intrinsics, self.H, self.W)

        poses = poses1.clone()
        # poses[:, :, 3] = poses[:, :, 3] * 0.33

        rays_o = rays['rays_o'].squeeze(0).cpu()
        rays_d = rays['rays_d'].squeeze(0).cpu()

        images = self.images[index] # [B, H, W, 3/4]
        C = images.shape[-1]

        if self.type == 'train' and not self.enable_semantic:
            images = torch.gather(images.view(-1, C), 0, torch.stack(C * [rays['inds'].cpu().squeeze(0)], -1)) # [B, N, 3/4]
        else:
            images = images.view(-1, C)

        if self.enable_semantic and self.type == 'train':
            if self.visualization:
                save_dir = f'zyq/project'
                Path(save_dir).parent.mkdir(exist_ok=True)
                Path(save_dir).mkdir(exist_ok=True)
                (Path(save_dir) / "sample").mkdir(exist_ok=True)

            directions = directions.view(self.H, self.W, 3)
            self.depth_scale = torch.abs(directions.cpu()[:, :, 2]) # z-axis's values

            selected_points = []
            self.sam_points = []

            num_depth_process_total = self.images.shape[0]
            self._labels = torch.zeros((self.H, self.W), dtype=torch.int)

            if self.num_depth_process % num_depth_process_total == 0:
                self.object_ids = []
                self.world_points = []
                self.select_origin = True
                # 测试一个点的投影影响
                # if self.num_depth_process == self._labels.shape[0]:
                    # return 'end'
                self.num_depth_process = 0
                self.object_id=0
                self.num_save = self.num_save + 1

            self.random_points = [torch.tensor([[self.H // 3, self.W // 3]])]
            H, W = self.H, self.W
            
            if self.select_origin == True:
                img = self.imgs[index].copy()

                self.num_depth_process = 0
                self.num_depth_process = self.num_depth_process + 1
                
                sam_feature = (self._sam_features[index]).to(self.device)
                self.predictor.set_feature(sam_feature, [H, W])
                in_labels = np.array([1])

                for random_point in self.random_points:
                    self.object_id = self.object_id + 1
                    self.object_ids.append(self.object_id)
                    visual = torch.zeros((H, W), dtype=torch.int).to(self.device) 
                    bool_tensor = torch.zeros((H, W), dtype=torch.int).bool()

                    random_point = random_point.flip(1)
                    masks, iou_preds, _ = self.predictor.predict(random_point.cpu().numpy(), in_labels, return_torch=True)
                    masks_max = masks[:, torch.argmax(iou_preds),:,:].squeeze(0)
                    masks_max = masks_max * ~bool_tensor.to(self.device)
                    # TODO: 这里需要考虑上面初始采样点的问题
                    if masks_max.nonzero().size(0) == 0: 
                        continue

                    sam_points_10 = torch.nonzero(masks_max)[torch.randint(high=torch.sum(masks_max), size=(10,))]
                    self.sam_points.append(sam_points_10)
                    sam_points_10 = sam_points_10.flip(1).cpu().numpy()

                    masks, iou_preds, _ = self.predictor.predict(sam_points_10, np.array([1]*sam_points_10.shape[0]), multimask_output=False, return_torch=True)
                    masks_max = masks[:, torch.argmax(iou_preds),:,:].squeeze(0)
                    masks_max = masks_max * ~bool_tensor.to(self.device)  


                    mask_size = masks_max.nonzero().size(0)
                    N_sample = min(int(self.N_total), mask_size)
                    selected_point = torch.nonzero(masks_max)[torch.randint(high=torch.sum(masks_max), size=(N_sample,))]
                    selected_points.append(selected_point)
                    self._labels[masks_max] = self.object_id
                    
                    depth_map = self.depths[index].view(self.H, self.W)
                    depth_map = (depth_map * self.depth_scale).numpy()
                    # depth_map = depth_map.numpy()


                    world_points_10 = []

                    for sam_point_10 in sam_points_10:
                        sam_point_10 = torch.tensor([sam_point_10]).flip(1)

                        x, y = sam_point_10[0, 1], sam_point_10[0, 0]
                        depth = depth_map[y, x]


                        K1 = self.intrinsics
                        K1 = np.array([[K1[0], 0, K1[2]],[0, K1[1], K1[3]],[0,0,1]])
                        E1 = np.array(poses.squeeze(0).cpu())
                        # E1 = np.stack([E1[:, 0], E1[:, 1]*-1, E1[:, 2]*-1, E1[:, 3]], 1)   # RUB -> RDF

                        # coordinates = np.column_stack((x, y, np.ones(10))).astype(int)
                        # pt_3d = (np.dot(np.linalg.inv(K1), coordinates.T)).T * depth[:, np.newaxis]
                        # pt_3d = np.concatenate((pt_3d, np.ones((10, 1), dtype=pt_3d.dtype)), axis=1)
                        
                        pt_3d = np.dot(np.linalg.inv(K1), np.array([x, y, 1])) * depth
                        pt_3d = np.append(pt_3d, 1)
                        # world_point = np.dot(np.concatenate((E1, [[0,0,0,1]]), 0), pt_3d)
                        world_point = np.dot(E1, pt_3d)

                        self.K1 = K1
                        world_points_10.append(world_point)
                        
                        if self.visualization:
                            

                            pt2 = (int(x), int(y))
                            radius = 5
                            color = (255, 0, 0)
                            thickness = 2
                            cv2.circle(img, pt2, radius, color, thickness)
                            
                            # image_cat = np.concatenate([img, rgb_array], axis=1)
                            # cv2.imwrite(f"{save_dir}/{self.object_id}_{self.metadata_items[idx].image_path.stem}.jpg", image_cat)
                            # print(f"{save_dir}/{self.metadata_items[idx].image_path.stem}.jpg")
                    self.world_points.append(world_points_10)
                    if self.visualization:
                        cv2.circle(img, (int(random_point[0,0]), int(random_point[0,1])), radius, (0, 0, 255), thickness)
                if self.visualization:
                    rgb_array = np.stack([self._labels.bool(), self._labels.bool(), self._labels.bool()], axis=2)* 255
                    image_cat = np.concatenate([img, rgb_array], axis=1)
                    cv2.imwrite(f"{save_dir}/{self.num_save}_{index}.jpg", image_cat)
                    
                self.select_origin = False  
        
            elif self.num_depth_process < num_depth_process_total:
                self.num_depth_process = self.num_depth_process + 1
                
                E2 = np.array(poses.squeeze(0).cpu())
                # E2 = np.stack([E2[:, 0], E2[:, 1]*-1, E2[:, 2]*-1, E2[:, 3]], 1)   # RUB -> RDF
                # w2c = np.linalg.inv(np.concatenate((E2, [[0,0,0,1]]), 0))
                w2c = np.linalg.inv(E2)

                if self.visualization:
                    img = self.imgs[index].copy()
                
                for point_idx in range(len(self.world_points)):
                    #得到投影点在该图像上的位置，随后进行采样
                    bool_tensor = torch.zeros((H, W), dtype=torch.int).bool()
                    sam_points_10 = []
                    for world_point in self.world_points[point_idx]:
                        pt_3d_trans = np.dot(w2c, world_point)
                        pt_2d_trans = np.dot(self.K1, pt_3d_trans[:3]) 
                        pt_2d_trans = pt_2d_trans / pt_2d_trans[2]
                        x2, y2 = int(pt_2d_trans[0]), int(pt_2d_trans[1])
                        if x2 >= 0 and x2 < self.W and y2 >= 0 and y2 < self.H:
                            depth_map2 = self.depths[index].view(self.H, self.W)
                            depth_map2 = (depth_map2 * self.depth_scale).numpy()
                            # depth_map2 = depth_map2.numpy()
                            depth2 = depth_map2[y2, x2]
                            depth_diff = np.abs(depth2 - pt_3d_trans[2])
                            pt2 = (int(pt_2d_trans[0]), int(pt_2d_trans[1]))
                            
                            if self.visualization:
                                #visualize
                                # print('Project inside image 2', self.depth, depth2, depth_diff)
                                radius = 5
                                color = (255, 0, 0)
                                thickness = 2
                                if depth_diff > occluded_threshold:
                                    color = (0, 255, 0)
                                    img = cv2.circle(img, pt2, radius, color, thickness)
                                    print(f'occluded points!!   the depth_diff: {depth_diff}')
                                    
                                    # cv2.imwrite(f"{save_dir}/{self.object_ids[point_idx]}_{self.metadata_items[idx].image_path.stem}_occluded_{depth_diff}.jpg", img)
                                    # print(f"{save_dir}/{self.metadata_items[idx].image_path.stem}_occluded_{depth_diff}.jpg")
                                    cv2.imwrite(f"{save_dir}/{self.num_save}_{index}.jpg", img)

                                else:
                                    img = cv2.circle(img, pt2, radius, color, thickness)
                                    # print(f"{save_dir}/{self.object_ids[point_idx]}_{self.metadata_items[idx].image_path.stem}_project.jpg   the depth_diff: {depth_diff}")
                                    cv2.imwrite(f"{save_dir}/{self.num_save}_{index}.jpg", img)

                            if depth_diff > occluded_threshold:
                                # print(f"{idx}   out of images")
                                continue
                            else:
                                sam_points_10.append(torch.tensor([[pt2[0],pt2[1]]]))
                                
                        else:
                            # print(f"{idx}   occluded")
                            continue
                    
                    
                    
                    if len(sam_points_10) == 0:
                        continue
                    else:
                        self.sam_points.append(sam_points_10)
                    
                    sam_points_10 = torch.cat(sam_points_10, 0)
                    #### SAM ###
                    in_labels = np.array([1])
                    H, W = self.H, self.W
                    sam_feature = (self._sam_features[index]).to(self.device)
                    self.predictor.set_feature(sam_feature, [H, W])
                            
                    
                    #从初始点采样 sam mask
                    if sam_points_10.shape[0]==1:
                        masks, iou_preds, _ = self.predictor.predict(sam_points_10.cpu().numpy(), np.array([1]*sam_points_10.shape[0]), multimask_output=True, return_torch=True)
                    else:
                        masks, iou_preds, _ = self.predictor.predict(sam_points_10.cpu().numpy(), np.array([1]*sam_points_10.shape[0]), multimask_output=False, return_torch=True)
                    masks_max = masks[:, torch.argmax(iou_preds),:,:].squeeze(0)
                    masks_max = masks_max * ~bool_tensor.to(self.device)
                    mask_size = masks_max.nonzero().size(0)
                    # TODO: 这里需要考虑上面初始采样点的问题
                    if masks_max.nonzero().size(0) == 0: 
                        continue
                    
                    N_sample = min(int(self.N_total), mask_size)
                    selected_point = torch.nonzero(masks_max)[torch.randint(high=torch.sum(masks_max), size=(N_sample,))]
                    selected_points.append(selected_point)
                    
                    self._labels[masks_max] = self.object_id

                if self.visualization and len(self.sam_points) > 0:
                    rgb_array = np.stack([self._labels.bool(), self._labels.bool(), self._labels.bool()], axis=2)* 255
                    image_cat = np.concatenate([img, rgb_array], axis=1)
                    print(f"{save_dir}/{index}.jpg")
                    cv2.imwrite(f"{save_dir}/{self.num_save}_{index}.jpg", image_cat)

            if len(selected_points) == 0:
                return None
            else:
                selected_points = torch.cat(selected_points)
            #从整张图像中采样
            bool_tensor = self._labels.bool().to(self.device)
            selected_points1_num = min(1024, torch.sum(~bool_tensor))
            selected_points1 = torch.nonzero(~bool_tensor)[torch.randint(high=torch.sum(~bool_tensor), size=(selected_points1_num,))]
            if selected_points1.size(0) == 0: 
                pass
            else:
                selected_points = torch.cat([selected_points, selected_points1], dim=0)
            
            near = torch.zeros((rays_o.shape[0],1), dtype=torch.int32)
            far = 10 * torch.ones((rays_o.shape[0],1), dtype=torch.int32)
            rays = torch.cat([rays_o, rays_d, near, far], dim=1)
            images = images.view(H, W, -1)[selected_points[:, 0], selected_points[:, 1]]
            rays = rays.view(H, W, -1)[selected_points[:, 0], selected_points[:, 1]]
            img_indices = index * torch.ones(images.shape[0], dtype=torch.int32)

            item = {
                    'rgbs': images,
                    'rays': rays, 
                    'img_indices': img_indices,
                    'labels': self._labels[selected_points[:, 0], selected_points[:, 1]],
                }
        else:
            near = torch.zeros((rays_o.shape[0],1), dtype=torch.int32)
            far = 10 * torch.ones((rays_o.shape[0],1), dtype=torch.int32)
            rays = torch.cat([rays_o, rays_d, near, far], dim=1)
            img_indices = index * torch.ones(rays_o.shape[0], dtype=torch.int32)

            item = {
                    'rgbs': images,
                    'rays': rays, 
                    'img_indices': img_indices,
                    # 'labels': self._labels[idx][selected_points[:, 0], selected_points[:, 1]],
                }

        
        return item
        
        
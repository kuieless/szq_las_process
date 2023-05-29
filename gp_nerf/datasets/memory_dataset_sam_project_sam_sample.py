from typing import List, Dict

import torch
from torch.utils.data import Dataset

from gp_nerf.datasets.dataset_utils import get_rgb_index_mask_sam_depth
from gp_nerf.image_metadata import ImageMetadata
from mega_nerf.misc_utils import main_tqdm, main_print
from mega_nerf.ray_utils import get_rays, get_ray_directions
from tools.segment_anything import sam_model_registry, SamPredictor

import numpy as np
import cv2
import sys
import glob
import os
from pathlib import Path
import random
import shutil

from tools.unetformer.uavid2rgb import remapping, custom2rgb


def init(device):
    sam_checkpoint = "tools/segment_anything/sam_vit_h_4b8939.pth"
    
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


class MemoryDataset_SAM(Dataset):

    def __init__(self, metadata_items: List[ImageMetadata], near: float, far: float, ray_altitude_range: List[float],
                 center_pixels: bool, device: torch.device, hparams):
        super(MemoryDataset_SAM, self).__init__()

        self.debug_one_images_sam = hparams.debug_one_images_sam
        print(f"train_on_1_val_corresponding_images: {hparams.debug_one_images_sam}")

        

        self.online_sam_label = hparams.online_sam_label
        self.visualization = False
        # sam
        self.device = device # device 'cpu'
        self.predictor = init(self.device)
        self.N_total = hparams.sample_ray_num
        # 假设所有图像的长宽一致
        self.W = metadata_items[0].W
        self.H = metadata_items[0].H
        self.sampling_mode = hparams.sampling_mode
        #get ray
        self.near = near
        self.far = far
        self.metadata_items = metadata_items
        self.ray_altitude_range = ray_altitude_range
        metadata_item=metadata_items[0]
        self.directions = get_ray_directions(metadata_item.W,
                                            metadata_item.H,
                                            metadata_item.intrinsics[0],
                                            metadata_item.intrinsics[1],
                                            metadata_item.intrinsics[2],
                                            metadata_item.intrinsics[3],
                                            center_pixels,
                                            device)

        # project
        self.select_origin = True
        self.num_depth_process = 0
        self.object_id=0
        self.depth_scale = torch.abs(self.directions.cpu()[:, :, 2]) # z-axis's values
        self.K1 = None
        self.world_point = None

        # rgbs = []
        rays = []
        indices = []
        labels = []
        sam_features = []
        is_vals =[]
        depths = []


        main_print('Loading data')
        if hparams.debug:
            metadata_items = metadata_items[:40]
        
        if self.debug_one_images_sam:
            used_files = []
            used_files.extend(glob.glob(os.path.join('/data/yuqi/code/GP-NeRF-semantic/zyq/val_000459_v4', '*.jpg')))
            used_files.extend(glob.glob(os.path.join('/data/yuqi/code/GP-NeRF-semantic/zyq/val_000459_v4/occluded', '*.jpg')))
            used_files.sort()
            use_index = []
            for used_file in used_files:
                use_index.append(int(Path(used_file).stem))
            use_index.sort()
            self.use_index = use_index
        
        for metadata_item in main_tqdm(metadata_items):
            image_data = get_rgb_index_mask_sam_depth(metadata_item)

            if image_data is None:
                continue
            
            #zyq : add labels
            image_rgbs, image_indices, image_keep_mask, label, sam_feature, is_val, depth = image_data

            indices.append(image_indices)
            labels.append(torch.zeros_like(label, dtype=torch.int))   #初始化一个tensor存储object id
            sam_features.append(sam_feature)
            is_vals.append(is_val)
            depths.append(depth)

        
        main_print('Finished loading data')

        self._img_indices = indices  #torch.stack(indices)
        self._labels = torch.stack(labels)
        self._sam_features = torch.stack(sam_features)
        self._is_vals = is_vals
        self._depths = torch.stack(depths)  # float16, N*(H*W)




    def __len__(self) -> int:
        # return self._rgbs.shape[0]
        return self._labels.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        occluded_threshold = 0.01
        
        if self._is_vals[idx]:
            print('is val')
            if self.num_depth_process % self._labels.shape[0] != 0:
                self.num_depth_process = self.num_depth_process + 1
            return None

        if self.visualization:
            save_dir = f'zyq/project'
            Path(save_dir).parent.mkdir(exist_ok=True)
            Path(save_dir).mkdir(exist_ok=True)

        if self.num_depth_process % self._labels.shape[0] == 0:
            self.select_origin = True
            # 测试一个点的投影影响
            if self.num_depth_process == self._labels.shape[0]:
                return 'end'
            self.num_depth_process = 0
              
        if self.select_origin == True:
            print('select one point to project')
            self.object_id = self.object_id + 1
            self.num_depth_process = self.num_depth_process + 1
            # 选择未被分配的点作为初始点，后续投影到其他图像上
            bool_tensor = self._labels[idx].bool()
            #投影的代码和其他的get item不同，下面的bool tensor要取 非
            random_point = torch.nonzero(~bool_tensor)[torch.randint(high=torch.sum(~bool_tensor), size=(1,))]
            
            depth_map = self._depths[idx].view(self.H, self.W)
            depth_map = (depth_map * self.depth_scale).numpy()
            # depth_map = depth_map.numpy()

            random_point[0, 1] = int(self.W * 2/5)
            random_point[0, 0] = int(self.H * 2/3)
            # random_point[0, 1] = int(self.W * 1/2)
            # random_point[0, 0] = int(self.H * 1/2)

            x, y = random_point[0, 1], random_point[0, 0]
            depth = depth_map[y, x]
            K1 = self.metadata_items[idx].intrinsics
            K1 = np.array([[K1[0], 0, K1[2]],[0, K1[1], K1[3]],[0,0,1]])
            E1 = np.array(self.metadata_items[idx].c2w)
            E1 = np.stack([E1[:, 0], E1[:, 1]*-1, E1[:, 2]*-1, E1[:, 3]], 1)
            pt_3d = np.dot(np.linalg.inv(K1), np.array([x, y, 1])) * depth
            pt_3d = np.append(pt_3d, 1)
            world_point = np.dot(np.concatenate((E1, [[0,0,0,1]]), 0), pt_3d)
            self.K1 = K1
            self.world_point = world_point
            random_point = random_point.flip(1)
            
            if self.visualization:
                rgb_array = np.stack([bool_tensor, bool_tensor, bool_tensor], axis=2)* 255

                img= cv2.imread(f"{str(self.metadata_items[idx].image_path)}")
                pt2 = (int(x), int(y))
                radius = 5
                color = (0, 0, 255)
                thickness = 2
                cv2.circle(img, pt2, radius, color, thickness)
                
                image_cat = np.concatenate([img, rgb_array], axis=1)
                cv2.imwrite(f"{save_dir}/{self.object_id}_{self.metadata_items[idx].image_path.stem}.jpg", image_cat)
                # print(f"{save_dir}/{self.metadata_items[idx].image_path.stem}.jpg")
                

        elif self.num_depth_process < self._labels.shape[0]:
            #得到投影点在该图像上的位置，随后进行采样
            bool_tensor = self._labels[idx].bool()
            self.num_depth_process = self.num_depth_process + 1
            E2 = np.array(self.metadata_items[idx].c2w)
            E2 = np.stack([E2[:, 0], E2[:, 1]*-1, E2[:, 2]*-1, E2[:, 3]], 1)
            w2c = np.linalg.inv(np.concatenate((E2, [[0,0,0,1]]), 0))
            pt_3d_trans = np.dot(w2c, self.world_point)
            pt_2d_trans = np.dot(self.K1, pt_3d_trans[:3]) 
            pt_2d_trans = pt_2d_trans / pt_2d_trans[2]

            x2, y2 = int(pt_2d_trans[0]), int(pt_2d_trans[1])
            if x2 >= 0 and x2 < self.W and y2 >= 0 and y2 < self.H:
                depth_map2 = self._depths[idx].view(self.H, self.W)
                depth_map2 = (depth_map2 * self.depth_scale).numpy()
                # depth_map2 = depth_map2.numpy()
                depth2 = depth_map2[y2, x2]
                depth_diff = np.abs(depth2 - pt_3d_trans[2])
                pt2 = (int(pt_2d_trans[0]), int(pt_2d_trans[1]))
                
                if self.visualization:
                    #visualize
                    img = cv2.imread(str(self.metadata_items[idx].image_path))
                    # print('Project inside image 2', self.depth, depth2, depth_diff)
                    radius = 5
                    color = (0, 0, 255)
                    thickness = 2
                    if depth_diff > occluded_threshold:
                        color = (0, 255, 0)
                        img = cv2.circle(img, pt2, radius, color, thickness)
                        print(f'occluded points!!   the depth_diff: {depth_diff}')
                        cv2.imwrite(f"{save_dir}/{self.metadata_items[idx].image_path.stem}_occluded_{depth_diff}.jpg", img)
                        print(f"{save_dir}/{self.metadata_items[idx].image_path.stem}_occluded_{depth_diff}.jpg")
                    else:
                        img = cv2.circle(img, pt2, radius, color, thickness)
                        cv2.imwrite(f"{save_dir}/{self.metadata_items[idx].image_path.stem}_project_{depth_diff}.jpg", img)
                        print(f"{save_dir}/{self.metadata_items[idx].image_path.stem}_project.jpg   the depth_diff: {depth_diff}")
                
                if depth_diff > occluded_threshold:
                    return None
                else:
                    random_point = torch.tensor([[pt2[0],pt2[1]]])
            else:
                print(f"{idx}   out of images")
                print()
                return None
          
        
        
        in_labels = np.array([1])
        H, W = self.H, self.W
        sam_feature = (self._sam_features[idx]).to(self.device)
        self.predictor.set_feature(sam_feature, [H, W])
                
        
        #visualize
        visual = torch.zeros((H, W), dtype=torch.int).to(self.device) 
        # visual_rgb = self.metadata_items[idx].load_image()
        # visual_rgb = np.flip(visual_rgb.numpy(), axis=2)
        visual_pseudo = np.flip(custom2rgb(remapping(self.metadata_items[idx].load_label().numpy())), axis=2)
        


        #从初始点采样 sam mask
        masks, iou_preds, _ = self.predictor.predict(random_point.cpu().numpy(), in_labels, return_torch=True)
        masks_max = masks[:, torch.argmax(iou_preds),:,:].squeeze(0)
        masks_max = masks_max * ~bool_tensor.to(self.device)
        mask_size = masks_max.nonzero().size(0)
        # TODO: 这里需要考虑上面初始采样点的问题
        if masks_max.nonzero().size(0) == 0: 
            return None
        else:
            self.select_origin = False
        
        N_sample = min(int(self.N_total), mask_size)
        selected_points = torch.nonzero(masks_max)[torch.randint(high=torch.sum(masks_max), size=(N_sample,))]
        candidate_tensor = torch.ones(selected_points.size(0),dtype=torch.int).to(self.device)
        selected_points_object_id = self.object_id * candidate_tensor
        
        self._labels[idx][masks_max] = self.object_id

        # *****************************************************************************
        if self.visualization:
            visual = masks_max
            visual = visual.cpu().numpy()
            rgb_array = np.stack([visual, visual, visual], axis=2)* 255
            
            for k in range(selected_points.size(0)):  # H, W
                x, y = selected_points[k, 0], selected_points[k, 1]
                if x < H and x >0 and y < W and y >0:
                    rgb_array[x, y] = np.array([0, 128, 0])

            image_cat = np.concatenate([img, visual_pseudo, rgb_array], axis=1)
            cv2.imwrite(f"{save_dir}/{self.object_id}_{self.metadata_items[idx].image_path.stem}_project_sample.jpg", image_cat)
            print(f"{save_dir}/{self.object_id}_{self.metadata_items[idx].image_path.stem}_project_sample.jpg")
        # *****************************************************************************


        image_rays = get_rays(self.directions, self.metadata_items[idx].c2w.to(self.device), self.near, self.far, self.ray_altitude_range)
        img_indices = self._img_indices[idx] * torch.ones(selected_points.shape[0], dtype=torch.int32)


        item = {
                # 'rgbs': self._rgbs[idx, selected_points[:, 0], selected_points[:, 1], :],
                'rays': image_rays[selected_points[:, 0], selected_points[:, 1], :], 
                'img_indices': img_indices,
                'labels': selected_points_object_id,
            }

        
        return item
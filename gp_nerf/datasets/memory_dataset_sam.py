from typing import List, Dict

import torch
from torch.utils.data import Dataset

from gp_nerf.datasets.dataset_utils import get_rgb_index_mask_sam
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




def init(device):
    # sam_checkpoint = "samnerf/segment_anything/sam_vit_h_4b8939.pth"
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

        # sam
        self.device = device # device 'cpu'
        self.predictor = init(self.device)
        self.N_total = hparams.sample_ray_num
        self.N_each = hparams.sam_sample_each
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

        # rgbs = []
        rays = []
        indices = []
        labels = []
        sam_features = []
        is_vals =[]

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
            image_data = get_rgb_index_mask_sam(metadata_item)

            if image_data is None:
                continue
            
            #zyq : add labels
            image_rgbs, image_indices, image_keep_mask, label, sam_feature, is_val = image_data

            indices.append(image_indices)
            labels.append(torch.tensor(label, dtype=torch.int))
            sam_features.append(sam_feature)
            is_vals.append(is_val)
        
        main_print('Finished loading data')

        self._img_indices = indices  #torch.stack(indices)
        self._labels = torch.stack(labels)
        self._sam_features = torch.stack(sam_features)
        self._is_vals = is_vals



    def __len__(self) -> int:
        # return self._rgbs.shape[0]
        return self._labels.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if self.debug_one_images_sam:
            if not (idx in self.use_index):
                random_number = random.randint(0, len(self.use_index))
                idx = random_number
        
        if self._is_vals[idx]:
            idx = idx + 1
            assert self._is_vals[idx] == False
        
        in_labels = np.array([1])
        H, W = self.H, self.W
        sam_feature = (self._sam_features[idx]).to(self.device)
        self.predictor.set_feature(sam_feature, [H, W])
        bool_tensor = torch.ones((H, W), dtype=torch.int).to(self.device)
        N_selected = 0
        selected_points = []
        selected_points_group = []
        group_id = 1  #  0 denotes unassigned
        selected_one = []
        #visualize
        visual = torch.zeros((H, W), dtype=torch.int).to(self.device) 
        save_dir = f'zyq/visual_sam_sample'

        # mode: 1 每次取一个点，识别一个mask，从中采样 
        if self.sampling_mode == 'per_mask':
            while N_selected < self.N_total:
                random_point = torch.nonzero(bool_tensor)[torch.randint(high=torch.sum(bool_tensor), size=(1,))]
                random_point = random_point.flip(1)
                selected_one.append(random_point)
                masks, iou_preds, _ = self.predictor.predict(random_point.cpu().numpy(), in_labels, return_torch=True)
                masks_max = masks[:, torch.argmax(iou_preds),:,:].squeeze(0)
                #在mask内选 N_each 个点，若不足 N_each ，则全选
                if masks_max.nonzero().size(0) ==0:
                    continue
                elif masks_max.nonzero().size(0) < self.N_each:
                    select_point = masks_max.nonzero()
                    selected_points.append(select_point)
                    # group_id_str = str(idx)+'_'+str(group_id)
                    # selected_points_group.append([group_id_str] * select_point.size(0))
                    group_id_record = 1000 * idx + group_id
                    selected_points_group.append(group_id_record * torch.ones(select_point.size(0)).to(self.device))
                    N_selected += masks_max.nonzero().size(0)
                else:
                    select_point = torch.nonzero(masks_max)[torch.randint(high=torch.sum(masks_max), size=(self.N_each,))]
                    selected_points.append(select_point)
                    group_id_record = 1000 * idx + group_id
                    selected_points_group.append(group_id_record * torch.ones(select_point.size(0),dtype=torch.int).to(self.device))
                    N_selected += self.N_each
                group_id += 1
                bool_tensor = bool_tensor * (~masks_max)
            
            selected_points = torch.cat(selected_points)
            selected_points_group = torch.cat(selected_points_group)
            assert selected_points.size(0) == selected_points_group.size(0)
            if selected_points.size(0) > self.N_total:
                selected_points = selected_points[:self.N_total]
                selected_points_group = selected_points_group[:self.N_total]

        elif self.sampling_mode == 'per_mask_threshold':
            while N_selected < self.N_total:
                random_point = torch.nonzero(bool_tensor)[torch.randint(high=torch.sum(bool_tensor), size=(1,))]
                random_point = random_point.flip(1)
                selected_one.append(random_point)
                masks, iou_preds, _ = self.predictor.predict(random_point.cpu().numpy(), in_labels, return_torch=True)
                masks_max = masks[:, torch.argmax(iou_preds),:,:].squeeze(0)
                images_size = H*W
                mask_size = masks_max.nonzero().size(0)
                if masks_max.nonzero().size(0) ==0:
                    continue
                if mask_size / images_size < 0.02:
                    N_sample = int(self.N_each*0.1)
                    select_point = torch.nonzero(masks_max)[torch.randint(high=torch.sum(masks_max), size=(N_sample,))]
                    selected_points.append(select_point)
                    group_id_record = 1000 * idx + group_id
                    selected_points_group.append(group_id_record * torch.ones(select_point.size(0),dtype=torch.int).to(self.device))
                    N_selected += N_sample
                elif mask_size / images_size < 0.05:
                    N_sample = int(self.N_each*0.2)
                    select_point = torch.nonzero(masks_max)[torch.randint(high=torch.sum(masks_max), size=(N_sample,))]
                    selected_points.append(select_point)
                    group_id_record = 1000 * idx + group_id
                    selected_points_group.append(group_id_record * torch.ones(select_point.size(0),dtype=torch.int).to(self.device))
                    N_selected += N_sample
                else:
                    N_sample = int(self.N_each*1)
                    select_point = torch.nonzero(masks_max)[torch.randint(high=torch.sum(masks_max), size=(N_sample,))]
                    selected_points.append(select_point)
                    group_id_record = 1000 * idx + group_id
                    selected_points_group.append(group_id_record * torch.ones(select_point.size(0),dtype=torch.int).to(self.device))
                    N_selected += N_sample
                group_id += 1
                bool_tensor = bool_tensor * (~masks_max)
                
                # *****************************************************************************
                #  #   # below is visualization to save the sampling images
            #     masks_max_vis = np.stack([masks_max.cpu().numpy(), masks_max.cpu().numpy(), masks_max.cpu().numpy()], axis=2)
            #     for point in selected_one:
            #         # print(point)
            #         for k in range(point.size(0)):  # W , H
            #             x, y = int(point[k, 0]), int(point[k, 1])
            #             for m in range(-2,2):
            #                 for n in range(-2,2):
            #                     if x+m < W and x+m >0 and y+n < H and y+n >0:
            #                         masks_max_vis[(y+n), (x+m)] = np.array([0, 0, 128])
            #   # cv2.imwrite(f"{save_dir}/mask_{idx}_{N_selected}.png", masks_max_vis.astype(np.int32)*255)
            #     visual += masks_max * group_id
            # visual = visual.cpu().numpy()
            # rgb_array = np.stack([visual, visual, visual], axis=2)*(127/visual.max()+127)
            # rgb_array_test = rgb_array
            # for point in selected_points:
            #     for k in range(point.size(0)):  # H, W
            #         x, y = point[k, 0], point[k, 1]
            #         if x < H and x >0 and y < W and y >0:
            #             rgb_array_test[x, y] = np.array([0, 128, 0])
            # for point in selected_one:
            #     # print(point)
            #     for k in range(point.size(0)):  # W , H
            #         x, y = int(point[k, 0]), int(point[k, 1])
            #         for m in range(-5,5):
            #             for n in range(-5,5):
            #                 if x+m < W and x+m >0 and y+n < H and y+n >0:
            #                     rgb_array_test[(y+n), (x+m)] = np.array([0, 0, 128])
            # cv2.imwrite(f"{save_dir}/mask_{idx}_final.png", (rgb_array_test).astype(np.int32))
            # print(f"{save_dir}/mask_{idx}_final.png")
            # *****************************************************************************
            selected_points = torch.cat(selected_points)
            selected_points_group = torch.cat(selected_points_group)
            assert selected_points.size(0) == selected_points_group.size(0)
            if selected_points.size(0) > self.N_total:
                selected_points = selected_points[:self.N_total]
                selected_points_group = selected_points_group[:self.N_total]
        
        # mode: 2  先随机采样所有点，然后从采样点中取一点，识别mask，划分group； 再拿未识别的点，重复上述过程
        
        elif self.sampling_mode == 'whole_image':
            selected_points = torch.ones_like(self._labels[idx])[torch.randint(high=torch.sum(self._labels[idx]), size=(self.N_total,))]
            selected_points_status = torch.ones_like()
            select_from_sampling = selected_points[torch.randint(high=torch.sum(selected_points), size=(1,))]
            select_point = torch.nonzero(masks_max)[torch.randint(high=torch.sum(masks_max), size=(self.N_each,))]

        else:
            raise ImportError
        
        metadata_item = self.metadata_items[idx]
        image_rays = get_rays(self.directions, metadata_item.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)
        img_indices = self._img_indices[idx] * torch.ones(selected_points.shape[0], dtype=torch.int32)
        
        item = {
            # 'rgbs': self._rgbs[idx, selected_points[:, 0], selected_points[:, 1], :],
            'rays': image_rays[selected_points[:, 0], selected_points[:, 1], :], 
            'img_indices': img_indices,
            'labels': self._labels[idx, selected_points[:, 0], selected_points[:, 1]].int(),
            'groups': selected_points_group,
            # 'selected_points': selected_points
        }
        return item

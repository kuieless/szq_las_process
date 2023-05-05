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

        # sam
        self.device = device # device 'cpu'
        self.predictor = init(self.device)
        self.N_total = hparams.sam_sample_total
        self.N_each = hparams.sam_sample_each
        # 假设所有图像的长宽一致
        self.W = metadata_items[0].W
        self.H = metadata_items[0].H

        rgbs = []
        rays = []
        indices = []
        labels = []
        sam_features = []

        main_print('Loading data')

        for metadata_item in main_tqdm(metadata_items):
            image_data = get_rgb_index_mask_sam(metadata_item)

            if image_data is None:
                continue
            
            #zyq : add labels
            image_rgbs, image_indices, image_keep_mask, label, sam_feature = image_data

            # print("image index: {}, fx: {}, fy: {}".format(metadata_item.image_index, metadata_item.intrinsics[0], metadata_item.intrinsics[1]))
            directions = get_ray_directions(metadata_item.W,
                                            metadata_item.H,
                                            metadata_item.intrinsics[0],
                                            metadata_item.intrinsics[1],
                                            metadata_item.intrinsics[2],
                                            metadata_item.intrinsics[3],
                                            center_pixels,
                                            device)
            image_rays = get_rays(directions, metadata_item.c2w.to(device), near, far, ray_altitude_range).cpu()  #.view(-1,8).cpu()

            rgbs.append(image_rgbs.float() / 255.)
            rays.append(image_rays)
            indices.append(image_indices)
            labels.append(torch.tensor(label, dtype=torch.int))
            sam_features.append(sam_feature)
        
        main_print('Finished loading data')

        self._rgbs = torch.stack(rgbs)
        self._rays = torch.stack(rays)
        self._img_indices = torch.stack(indices)
        self._labels = torch.stack(labels)
        self._sam_features = torch.stack(sam_features)

    def __len__(self) -> int:
        return self._rgbs.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        #sam_mask
        # 加入sam的训练去掉了val dataset
        in_labels = np.array([1])
        H, W = self.H, self.W
        sam_feature = (self._sam_features[idx]).to(self.device)
        self.predictor.set_feature(sam_feature, [H, W])
        bool_tensor = torch.ones((H, W), dtype=torch.int).to(self.device)
        N_selected = 0
        selected_points = []
        selected_points_group = []
        group_id = 0
        selected_one = []
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
                selected_points_group.append(group_id * torch.ones(select_point.size(0)).to(self.device))
                N_selected += masks_max.nonzero().size(0)
            else:
                select_point = torch.nonzero(masks_max)[torch.randint(high=torch.sum(masks_max), size=(self.N_each,))]
                selected_points.append(select_point)
                selected_points_group.append(group_id * torch.ones(select_point.size(0)).to(self.device))
                N_selected += self.N_each
            group_id += 1
            bool_tensor = bool_tensor * (~masks_max)
        
        selected_points = torch.cat(selected_points)
        selected_points_group = torch.cat(selected_points_group)
        assert selected_points.size(0) == selected_points_group.size(0)
        if selected_points.size(0) > self.N_total:
            selected_points = selected_points[:self.N_total]
            selected_points_group = selected_points_group[:self.N_total]


        return {
            'rgbs': self._rgbs[0, selected_points[:, 0], selected_points[:, 1], :],
            'rays': self._rays[0, selected_points[:, 0], selected_points[:, 1], :],
            'img_indices': self._img_indices[0, selected_points[:, 0], selected_points[:, 1]],
            'labels': self._labels[0, selected_points[:, 0], selected_points[:, 1]],
            'groups': selected_points_group,
        }

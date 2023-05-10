"""
Support depth only
"""
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from gp_nerf.datasets.dataset_utils import get_rgb_index_mask_monocues
from gp_nerf.image_metadata import ImageMetadata
from mega_nerf.misc_utils import main_tqdm, main_print
from mega_nerf.ray_utils import get_rays, get_ray_directions

import numpy as np
import random

class MemoryDataset(Dataset):

    def __init__(self, metadata_items: List[ImageMetadata], near: float, far: float, ray_altitude_range: List[float],
                 center_pixels: bool, device: torch.device, hparams):
        super(MemoryDataset, self).__init__()

        self.sample_random_num_each = hparams.sample_random_num_each

        self.metadata_items = metadata_items
        self.near = near
        self.far = far
        self.ray_altitude_range = ray_altitude_range
        self._c2ws = torch.cat([x.c2w.unsqueeze(0) for x in metadata_items])
        self.device = device

        rgbs = []
        indices = []
        labels = []
        depths = []

        all_rays = []
        all_img_indices = []
        all_rgbs = []
        all_labels = []
        main_print('Loading data')

        metadata_item = metadata_items[0]
        self.total_pixels = metadata_item.H * metadata_item.W
        self.sampling_size = hparams.sample_ray_num
        self.W = metadata_items[0].W
        self.H = metadata_items[0].H
        self._directions = get_ray_directions(metadata_item.W,
                                        metadata_item.H,
                                        metadata_item.intrinsics[0],
                                        metadata_item.intrinsics[1],
                                        metadata_item.intrinsics[2],
                                        metadata_item.intrinsics[3],
                                        center_pixels,
                                        device)

        if hparams.debug:
            metadata_items = metadata_items[:40]

        for metadata_item in main_tqdm(metadata_items):
            image_data = get_rgb_index_mask_monocues(metadata_item)

            if image_data is None:
                continue
            
            #zyq : add labels
            image_rgbs, image_indices, image_keep_mask, label, normal, depth = image_data
            #print(image_rgbs.shape, metadata_item.image_path)

            #get rays
            image_rays = get_rays(self._directions, metadata_item.c2w.to(device), near, far, ray_altitude_range).view(-1,8).cpu()
            if image_keep_mask is not None:
                image_rays = image_rays[image_keep_mask == True]
            all_rays.append(image_rays[::10])
            all_img_indices.append(image_indices * torch.ones(image_rays.shape[0], dtype=torch.int32)[::10])
            all_rgbs.append(image_rgbs[::10])
            all_labels.append(label.int()[::10])

            
            
            rgbs.append(image_rgbs)
            indices.append(image_indices)
            labels.append(label.int())
            depths.append(depth)
        main_print('Finished loading data')

        self._all_rgbs = torch.cat(all_rgbs)
        self._all_rays = torch.cat(all_rays)
        self._all_img_indices = torch.cat(all_img_indices)
        self._all_labels = torch.cat(all_labels)

        self._rgbs = rgbs # torch.stack(rgbs) # N*(H*W)*3, Byte/uint8
        self._depths = depths #torch.stack(depths)  # float16, N*(H*W)
        self._img_indices = indices # N, int
        self._labels = labels # torch.stack(labels) # Byte/uint8, N*H*W*1
        
        print('self.rgbs', self._rgbs[0].shape, self._rgbs[0].dtype)
        if self._depths[0] is not None:
            print('self.depths', self._depths[0].shape, self._depths[0].dtype)
        print('self._img_indices', len(self._img_indices))
        print('self._labels', self._labels[0].shape, self._labels[0].dtype)
        self.all_sampling_idx = torch.randperm(self._all_rgbs.shape[0])

    def __len__(self) -> int:
        return len(self._rgbs)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # print(idx)
        total_pixels = self._rgbs[idx].shape[0]
        sampling_idx = torch.randperm(total_pixels)[:self.sampling_size]

        metadata_item = self.metadata_items[idx]
        if metadata_item.is_val:
            directions = self._directions[:, :self.W // 2].contiguous()
            #print(idx, 'val')
        else:
            directions = self._directions
        directions = directions.view(-1, 3)[sampling_idx]
        depth_scale = torch.abs(directions[:, 2])

        image_rays = get_rays(directions, metadata_item.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)

        rays = image_rays.view(-1,8).cpu()

        img_indices = self._img_indices[idx] * torch.ones(rays.shape[0], dtype=torch.int32)
        
        # get random sampling
        random_number = random.randint(0, self._all_rgbs.shape[0]-self.sample_random_num_each)
        all_sampling_idx = self.all_sampling_idx[random_number:random_number+self.sample_random_num_each]

        item = {
            'rgbs': self._rgbs[idx][sampling_idx].float() / 255.,
            'rays': rays,
            'img_indices': img_indices,
            'labels': self._labels[idx][sampling_idx].int(),
        }

        item['random_rgbs'] = self._all_rgbs[all_sampling_idx].float() / 255.
        item['random_rays'] = self._all_rays[all_sampling_idx]
        item['random_img_indices'] = self._all_img_indices[all_sampling_idx]
        item['random_labels'] = self._all_labels[all_sampling_idx].int()

        if self._depths[0] is not None:
            item['depths'] = self._depths[idx][sampling_idx].float()
            item['depth_scale'] = depth_scale
        # print(item['rays'].size())
        
        
        return item

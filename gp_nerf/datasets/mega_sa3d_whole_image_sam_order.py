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


class MemoryDataset_SAM_sa3d(Dataset):

    def __init__(self, metadata_items: List[ImageMetadata], near: float, far: float, ray_altitude_range: List[float],
                 center_pixels: bool, device: torch.device, hparams, predictor=None):
        super(MemoryDataset_SAM_sa3d, self).__init__()

        self.debug_one_images_sam = hparams.debug_one_images_sam
        print(f"train_on_1_val_corresponding_images: {hparams.debug_one_images_sam}")

        
        self.num_semantic_classes = hparams.num_semantic_classes
        self.online_sam_label = hparams.online_sam_label
        self.visualization = False
        
        # sam
        self.device = device # device 'cpu'
        self.predictor = init(self.device)
        self.N_total = int(hparams.sample_ray_num / 5)
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
        
        # metadata_items = metadata_items[27:]

        main_print('Loading data')
        if hparams.debug:
            metadata_items = metadata_items[:70]
        
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
        
        #27是初始祯
        if idx < 27:
            return None
        
        occluded_threshold = 0.01
        num_random_points = 5
        selected_points = []
        self.random_points = []
        self.num_save=0
        
        self.sam_points = []
        
        if self._is_vals[idx]:
            print('is val')
            if self.num_depth_process % self._labels.shape[0] != 0:  # TODO: 这里要考虑building的第一张图片为val
                self.num_depth_process = self.num_depth_process + 1
            return None

        if self.visualization:
            save_dir = f'zyq/project'
            Path(save_dir).parent.mkdir(exist_ok=True)
            Path(save_dir).mkdir(exist_ok=True)
            (Path(save_dir) / "sample").mkdir(exist_ok=True)


        if self.num_depth_process % self._labels.shape[0] == 0:
            self.object_ids = []
            self.world_points = []
            self.select_origin = True
            # 测试一个点的投影影响
            if self.num_depth_process == self._labels.shape[0]:
                return 'end'
            self._labels = torch.zeros_like(self._labels)
            self.num_depth_process = 0
            self.object_id=0
        
        
        sam_feature = (self._sam_features[idx]).to(self.device)
        depth_map = self._depths[idx].view(self.H, self.W)
        depth_map = (depth_map * self.depth_scale).numpy()

        if self.select_origin == True:
            labels = []
            self.num_depth_process = 27
            self.num_depth_process = self.num_depth_process + 1

            if self.visualization:
                img= cv2.imread(f"{str(self.metadata_items[idx].image_path)}")


            self.select_origin = False   
            labels = np.load('/data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/residence_000027_sam_order.npy')
            # labels=torch.HalfTensor(labels).view(-1, labels.shape[-1])[:,90:90+self.num_semantic_classes]
            labels=torch.HalfTensor(labels).view(-1, labels.shape[-1])[:,:self.num_semantic_classes]

            # labels = torch.stack(labels, dim=1)

            if self.visualization and idx == 27:

                save_dir = f'zyq/project'
                Path(save_dir).parent.mkdir(exist_ok=True)
                Path(save_dir).mkdir(exist_ok=True)
                (Path(save_dir) / "sample").mkdir(exist_ok=True)
                rgb_array = np.stack([self._labels[idx].bool(), self._labels[idx].bool(), self._labels[idx].bool()], axis=2)* 255
                image_cat = np.concatenate([img, rgb_array], axis=1)
                cv2.imwrite(f"{save_dir}/first_{self.num_save}.jpg", image_cat)
                self.num_save = self.num_save + 1

        elif self.num_depth_process < self._labels.shape[0]:
            self.num_depth_process = self.num_depth_process + 1
            # if idx < 58:
            #     return None
            labels = None
        
        
        image_rays = get_rays(self.directions, self.metadata_items[idx].c2w.to(self.device), self.near, self.far, self.ray_altitude_range)
        img_indices = self._img_indices[idx] * torch.ones(self.H*self.W, dtype=torch.int32)


        item = {
                # 'rgbs': self._rgbs,
                'rays': image_rays.view(-1, 8), 
                'img_indices': img_indices,
            }
        if labels is not None:
            item['labels'] = labels
        
        item['depth'] = torch.tensor(depth_map)
        item['sam_feature'] = sam_feature.cpu()

        return item
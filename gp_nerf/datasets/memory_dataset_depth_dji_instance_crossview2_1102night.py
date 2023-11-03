from typing import List, Dict

import torch
from torch.utils.data import Dataset

from gp_nerf.datasets.dataset_utils import get_rgb_index_mask_depth_dji_instance_crossview
from gp_nerf.image_metadata import ImageMetadata
from mega_nerf.misc_utils import main_tqdm, main_print
from mega_nerf.ray_utils import get_rays, get_ray_directions

import numpy as np
import glob
import os
from pathlib import Path
import random
from PIL import Image
import cv2

class MemoryDataset(Dataset):

    def __init__(self, metadata_items: List[ImageMetadata], near: float, far: float, ray_altitude_range: List[float],
                 center_pixels: bool, device: torch.device, hparams=None):
        super(MemoryDataset, self).__init__()
        self.hparams = hparams
        rgbs = []
        rays = []
        indices = []
        instances = []
        instances_crossview = []
        self.metadata_items = metadata_items

        depth_djis = []
        depth_scales = []
        metadata_item = metadata_items[0]

        self.W = metadata_item.W
        self.H = metadata_item.H
        
        self._directions = get_ray_directions(metadata_item.W,
                                        metadata_item.H,
                                        metadata_item.intrinsics[0],
                                        metadata_item.intrinsics[1],
                                        metadata_item.intrinsics[2],
                                        metadata_item.intrinsics[3],
                                        center_pixels,
                                        device)
        self._depth_scale = torch.abs(self._directions[:, :, 2]).view(-1).cpu()
        
        main_print('Loading data')
        if hparams.debug:
            metadata_items = metadata_items[::20]
        metadata_items = metadata_items[207:208]
        load_subset = 0
        for metadata_item in main_tqdm(metadata_items):
        # for metadata_item in main_tqdm(metadata_items[:40]):
            if hparams.enable_semantic and metadata_item.is_val:  # 训练语义的时候要去掉val图像
                continue
            if hparams.use_subset:
                used_files = []
                for ext in ('*.png', '*.jpg'):
                    used_files.extend(glob.glob(os.path.join(f'{hparams.dataset_path}/subset/rgbs', ext)))
                    # used_files.extend(glob.glob(os.path.join('/data/yuqi/Datasets/DJI/Yingrenshi_20230926_subset/train/rgbs', ext)))
                    # used_files.extend(glob.glob(os.path.join('/data/yuqi/Datasets/DJI/Yingrenshi_20230926_subset/val/rgbs', ext)))
                used_files.sort()
                file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in used_files]
                if (metadata_item.label_path == None): #增强label集yingrenshi上只做了subset，所以可能会没有
                    continue
                else:
                    if (Path(metadata_item.label_path).stem not in file_names):
                        continue
                    else:
                        load_subset = load_subset+1

            image_data = get_rgb_index_mask_depth_dji_instance_crossview(metadata_item)

            if image_data is None:
                continue
            #改成读取instance label
            image_rgbs, image_indices, image_keep_mask, instance, depth_dji, instance_crossview = image_data
            
            image_rays = get_rays(self._directions, metadata_item.c2w.to(device), near, far, ray_altitude_range).view(-1, 8).cpu()
            
            depth_scale = self._depth_scale
            if image_keep_mask is not None:
                image_rays = image_rays[image_keep_mask == True]
                depth_scale = depth_scale[image_keep_mask == True]


            rgbs.append(image_rgbs)
            rays.append(image_rays)
            indices.append(image_indices)
            instances.append(torch.tensor(instance, dtype=torch.int))
            instances_crossview.append(torch.tensor(instance_crossview, dtype=torch.int))
            depth_djis.append(depth_dji / depth_scale)

        print(f"load_subset: {load_subset}")
        main_print('Finished loading data')

        self._rgbs = rgbs
        self._rays = rays
        self._img_indices = indices  #  这个代码只存了一个
        self._instances = instances
        self._instance_crossview = instances_crossview

        self._depth_djis = depth_djis 

    def __len__(self) -> int:
        return len(self._rgbs)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        visualization = True

        metadata_current = self.metadata_items[self._img_indices[idx]]
        
        if int(Path(metadata_current.image_path).stem) != 207:
            return None
        
        # 先用cross view对 instance进行一个并集判断
        sample_dict = {}
        instance = self._instances[idx].clone().detach()
        instance_crossview = self._instance_crossview[idx].clone().detach()


        unique_labels,counts = torch.unique(instance, return_counts=True)
        non_zero_indices = unique_labels != 0
        unique_labels = unique_labels[non_zero_indices]
        counts = counts[non_zero_indices]
        
        for uni in unique_labels:
            uni_mask = instance==uni
            label_in_crossview = instance_crossview[uni_mask]
            unique_label_in_crossview,counts_label_in_crossview = torch.unique(label_in_crossview, return_counts=True)
            non_zero_indices = (unique_label_in_crossview != 0)
            unique_label_in_crossview = unique_label_in_crossview[non_zero_indices]
            counts_label_in_crossview = counts_label_in_crossview[non_zero_indices]
            if counts_label_in_crossview.shape[0] == 0:
                continue
            max_count_index = torch.argmax(counts_label_in_crossview)
            most_frequent_label = unique_label_in_crossview[max_count_index]
            if uni == 11913:
                a = 1
            ##### 2023 1102  用most_frequent_label组成dict
            if uni_mask.nonzero().shape[0] > 0.0005 * self.H * self.W:
                
                if counts_label_in_crossview[max_count_index]/label_in_crossview.shape[0] < 0.5:
                    continue
                if f'{most_frequent_label}' not in sample_dict:
                    sample_dict[f'{most_frequent_label}'] = []
                each_mask = torch.zeros_like(instance)
                each_mask[uni_mask] = instance[uni_mask]
                sample_dict[f'{most_frequent_label}'].append(each_mask)

            #####  2023 1101 把异常值剔掉
            # if uni != most_frequent_label:
            #     instance[uni_mask] = 0
        
        ## 在每个块中选随机选一个
        selected_tensors = {}
        for key, tensor_list in sample_dict.items():
            # if key != '11989':
                # continue
            if tensor_list != []:
                print(key)
                random_index = np.random.randint(0, len(tensor_list))  # 使用NumPy生成随机索引
                selected_tensors[key] = tensor_list[random_index]

        tensor_list = list(selected_tensors.values())
        stacked_tensors = torch.stack(tensor_list)
        instance_new = torch.sum(stacked_tensors, dim=0)
        
        if visualization:
            def label_to_color(label_tensor):
                unique_labels = torch.unique(label_tensor)
                color_image = torch.zeros(label_tensor.shape[0], label_tensor.shape[1],3).to(torch.uint8)

                for i in unique_labels:
                    if i ==0:
                        continue
                    color_image[label_tensor==i]=torch.randint(0, 256, (3,), dtype=torch.uint8)
                return color_image
            
            color = label_to_color(self._instances[idx].long().view(self.H, self.W)) *0.7 + 0.3 * self._rgbs[idx].clone().view(self.H, self.W, 3)
            color_crossview = label_to_color(instance_crossview.long().view(self.H, self.W)) *0.7 + 0.3 * self._rgbs[idx].clone().view(self.H, self.W, 3)
            results = label_to_color(instance_new.long().view(self.H, self.W)) *0.7 + 0.3 * self._rgbs[idx].clone().view(self.H, self.W, 3)
            vis_img = np.concatenate([color.cpu().numpy(), color_crossview.cpu().numpy(), results.cpu().numpy()], axis=1)
            Path(f"zyq/1102_crossview_train/viz").mkdir(exist_ok=True, parents=True)
            cv2.imwrite(f"zyq/1102_crossview_train/viz/{self._img_indices[idx]}_{random.randint(1, 100)}.jpg", vis_img)
        

        # Path(f"zyq/1102_crossview_train/results").mkdir(exist_ok=True, parents=True)
        # Image.fromarray(instance.view(self.H, self.W).cpu().numpy().astype(np.uint32)).save(f"zyq/1102_crossview_train/results/%06d.png" % (self._img_indices[idx]))
        

        if idx == len(self._rgbs) - 1:
            # torch.cuda.empty_cache()
            # return 'end'
            return None

        else:
            # torch.cuda.empty_cache()
            print(f"process idx : {self._img_indices[idx]}")
            return None
        
        
        # 找到非零值的索引
        nonzero_indices = torch.nonzero(instance).squeeze()


        
        sampling_idx = nonzero_indices[torch.randperm(nonzero_indices.size(0))[:self.hparams.batch_size]]
        #### 1. 若点不够，返回None,但这个在多个batch size会出问题
        # if sampling_idx.shape[0] == 0 or sampling_idx.shape[0] < self.hparams.batch_size:
        #     return None
        #### 2. 点不够， 则换张图采样
        if sampling_idx.shape[0] == 0 or sampling_idx.shape[0] < self.hparams.batch_size:

            index_shuffle= list(range(len(self._rgbs)))
            index_shuffle.remove(idx)  # 从列表中移除已知的索引

            # 从剩下的数字中随机选择一个数
            next_idx = random.choice(index_shuffle)
            item = self.__getitem__(next_idx)
            return item



        item = {
            'rgbs': self._rgbs[idx][sampling_idx].float() / 255.,
            'rays': self._rays[idx][sampling_idx],
            'img_indices': self._img_indices[idx] * torch.ones(sampling_idx.shape[0], dtype=torch.int32),
            # 'labels': self._instances[idx][sampling_idx].int(),
            'labels': instance[sampling_idx],
        }
        if self._depth_djis[idx] is not None:
            item['depth_dji'] = self._depth_djis[idx][sampling_idx]
            # item['depth_scale'] = self._depth_scale[sampling_idx]  # 10.23  这里有个bug，   depth_scale没经过mask过滤
        
        return item
    

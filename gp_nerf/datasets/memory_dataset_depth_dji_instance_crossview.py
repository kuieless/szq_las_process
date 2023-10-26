from typing import List, Dict

import torch
from torch.utils.data import Dataset

from gp_nerf.datasets.dataset_utils import get_rgb_index_mask_depth_dji_instance
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
        depth_djis = []
        depth_scales = []
        metadata_item = metadata_items[0]
        self.metadata_items =metadata_items

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
        depth_scale_full = torch.abs(self._directions[:, :, 2]).view(-1).cpu()
        
        main_print('Loading data')
        if hparams.debug:
            metadata_items = metadata_items[::20]
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

            image_data = get_rgb_index_mask_depth_dji_instance(metadata_item)

            if image_data is None:
                continue
            #改成读取instance label
            image_rgbs, image_indices, image_keep_mask, instance, depth_dji = image_data
            
            image_rays = get_rays(self._directions, metadata_item.c2w.to(device), near, far, ray_altitude_range).view(-1, 8).cpu()
            
            depth_scale = depth_scale_full.clone()
            if image_keep_mask is not None:
                image_rays = image_rays[image_keep_mask == True]
                depth_scale = depth_scale[image_keep_mask == True]


            rgbs.append(image_rgbs)
            rays.append(image_rays)
            indices.append(image_indices)
            instances.append(torch.tensor(instance, dtype=torch.int))
            depth_djis.append(depth_dji / depth_scale)
            depth_scales.append(depth_scale)
        print(f"load_subset: {load_subset}")
        main_print('Finished loading data')

        self._rgbs = rgbs
        self._rays = rays
        self._img_indices = indices  #  这个代码只存了一个
        self._labels = instances
        self._depth_djis = depth_djis 
        self._depth_scales = depth_scales 


    def __len__(self) -> int:
        return len(self._rgbs)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        


        # 找到第一张图的非零值的索引, 从非零值的索引中随机采样
        nonzero_indices = torch.nonzero(self._labels[idx]).squeeze()
        sampling_idx_current = nonzero_indices[torch.randperm(nonzero_indices.size(0))[:self.hparams.batch_size//2]]
        if sampling_idx_current.shape[0] ==0:
            return None

        ###可视化采样点
        # visualization = True
        # if visualization:
        #     save_dir = f'zyq/1026_cross_view'
        #     Path(save_dir).parent.mkdir(exist_ok=True)
        #     Path(save_dir).mkdir(exist_ok=True)
        #     vis_img = torch.zeros_like(self._rgbs[idx])
        #     vis_img[sampling_idx, :] = torch.tensor([255,0,0]).to(torch.uint8)
        #     cv2.imwrite(f"{save_dir}/{idx}.png", vis_img.view(self.H,self.W,3).cpu().numpy())


        # ### NOTE 投影需要用的是z分量的depth, 应人石数据没有考虑左右图片，暂时先按照（H,W）进行投影
        labels_current = self._labels[idx].view(self.H, self.W)
        labels_current_unique = labels_current.unique()
        
        metadata_current = self.metadata_items[self._img_indices[idx]]
        
        # 将当前的点根据深度生成点云

        depth_map = (self._depth_djis[idx] * self._depth_scales[idx])
        inf_mask = torch.isinf(depth_map)
        depth_map[inf_mask] = depth_map[~inf_mask].max()


        x_grid, y_grid = torch.meshgrid(torch.arange(self.W), torch.arange(self.H))
        x_grid, y_grid = x_grid.T.flatten(), y_grid.T.flatten()

        pixel_coordinates = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)
        K1 = metadata_current.intrinsics
        K1 = torch.tensor([[K1[0], 0, K1[2]], [0, K1[1], K1[3]], [0, 0, 1]])
        pt_3d = depth_map[:, None] * (torch.linalg.inv(K1) @ pixel_coordinates[:, :, None].float()).squeeze()
        arr2 = torch.ones((pt_3d.shape[0], 1))
        pt_3d = torch.cat([pt_3d, arr2], dim=-1)
        # pt_3d = pt_3d[valid_depth_mask]
        pt_3d = pt_3d.view(-1, 4)
        E1 = metadata_current.c2w.clone().detach()
        E1 = torch.stack([E1[:, 0], -E1[:, 1], -E1[:, 2], E1[:, 3]], dim=1)
        world_point = torch.mm(torch.cat([E1, torch.tensor([[0, 0, 0, 1]])], dim=0), pt_3d.t()).t()
        world_point = world_point[:, :3] / world_point[:, 3:4]




        success = 0
        index_shuffle= list(range(len(self._rgbs)))
        random.shuffle(index_shuffle)
        index_count = 0
        ## 投影到下一张图片
        while index_count < len(self._rgbs):
            
            next_idx = index_shuffle[index_count]
            if next_idx == idx:
                continue

            metadata_next = self.metadata_items[self._img_indices[next_idx]]
            E2 = metadata_next.c2w.clone().detach()
            E2 = torch.stack([E2[:, 0], E2[:, 1]*-1, E2[:, 2]*-1, E2[:, 3]], 1)
            w2c = torch.inverse(torch.cat((E2, torch.tensor([[0, 0, 0, 1]])), dim=0))
            points_homogeneous = torch.cat((world_point, torch.ones((world_point.shape[0], 1), dtype=torch.float32)), dim=1)
            pt_3d_trans = torch.mm(w2c, points_homogeneous.t())
            pt_2d_trans = torch.mm(K1, pt_3d_trans[:3])
            pt_2d_trans = pt_2d_trans / pt_2d_trans[2]
            projected_points = pt_2d_trans[:2].t()

            


            ########### 考虑遮挡
            threshold= 0.02
            image_width, image_height = self.W, self.H
            mask_x = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < image_width)
            mask_y = (projected_points[:, 1] >= 0) & (projected_points[:, 1] < image_height)
            mask = mask_x & mask_y

            ### NOTE:  第二张图的depth
            depth_next = (self._depth_djis[next_idx] * self._depth_scales[next_idx]).view(self.H,self.W)


            x = projected_points[:, 0].long()
            y = projected_points[:, 1].long()
            x[~mask] = 0
            y[~mask] = 0
            depth_map_next = depth_next[y, x]
            depth_map_next[~mask] = -1e6

            depth_z = pt_3d_trans[2]
            mask_z = depth_z < (depth_map_next + threshold)
            mask_xyz = (mask & mask_z)

            # 从第二张图中拿到投影得到的label
            labels_next = self._labels[next_idx].view(self.H,self.W)
            # 将投影成功的label拿到

            # 找到第一张图中每个label的对应位置，再根据 flatten 的顺序 找到它投影在第二张图上的位置
            # 接下来用投影得到的label，把cross view的标签一致，并且从两张图中分别采样
            ### SAM得到的mask可以对id进行递增，重新排序 tools/segment_anything/helpers/1019_get_sammask_autogenerate.py 
            # 这里先用 ground truth instance label 进行处理
            for uni  in labels_current_unique:
                if uni == 0:
                    continue
                first_index = (labels_current == uni).flatten()
                first_index_valid = first_index * mask_xyz
                if first_index_valid.sum() == 0:
                    continue
                project_label = labels_next[y[first_index_valid], x[first_index_valid]]
                unique_values, counts = torch.unique(project_label, return_counts=True)
                max_count_index = torch.argmax(counts)
                if unique_values[max_count_index]==0:
                    continue
                labels_next[labels_next==unique_values[max_count_index]] = uni  # 这里把第二张图和第一张图的对应标签变为一致
                success += 1
            
            if success>0:
                break

            index_count += 1
        

        # 找到第二张图的非零值的索引, 从非零值的索引中随机采样
        nonzero_indices = torch.nonzero(self._labels[next_idx]).squeeze()
        sampling_idx_next = nonzero_indices[torch.randperm(nonzero_indices.size(0))[:self.hparams.batch_size//2]]
                
        rgbs = torch.cat([self._rgbs[idx][sampling_idx_current].float() / 255., 
                          self._rgbs[next_idx][sampling_idx_next].float() / 255.], dim=0)
        rays = torch.cat([self._rays[idx][sampling_idx_current], 
                          self._rays[next_idx][sampling_idx_next]], dim=0)
        img_indices = torch.cat([self._img_indices[idx] * torch.ones(sampling_idx_current.shape[0], dtype=torch.int32), 
                                 self._img_indices[next_idx] * torch.ones(sampling_idx_next.shape[0], dtype=torch.int32)], dim=0)
        labels = torch.cat([self._labels[idx][sampling_idx_current].int(), 
                            self._labels[next_idx][sampling_idx_next].int()], dim=0)
        depth_dji = torch.cat([self._depth_djis[idx][sampling_idx_current], 
                          self._depth_djis[next_idx][sampling_idx_next]], dim=0)
        

        item = {
            'rgbs': rgbs,
            'rays': rays,
            'img_indices': img_indices,
            'labels': labels,
            'depth_dji': depth_dji,
        }
        
        return item
    

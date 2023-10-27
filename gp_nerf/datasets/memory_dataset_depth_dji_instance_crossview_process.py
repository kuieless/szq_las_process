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
            metadata_items = metadata_items[::10]
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
        ###NOTE 需要将shuffle调成False, 不打乱，按照顺序处理

        # 拿到当前图像的数据
        img_current = self._rgbs[idx].clone().view(self.H, self.W, 3)
        instances_current = self._labels[idx].clone().view(self.H, self.W)
        depth_current = (self._depth_djis[idx] * self._depth_scales[idx]).view(self.H, self.W)
        metadata_current = self.metadata_items[self._img_indices[idx]]
        visualization = True
        if visualization:
            color_current = torch.zeros_like(img_current)
            unique_label = torch.unique(instances_current)
            for uni in unique_label:
                if uni ==0:
                    continue
                random_color = torch.randint(0, 256, (3,), dtype=torch.uint8)
                if (instances_current==uni).sum() != 0:
                    color_current[instances_current==uni,:] = random_color
            vis_img1 = 0.7 * color_current + 0.3 * img_current



        index_list= list(range(len(self._rgbs)))
        index_list.remove(idx)  # 从列表中移除已知的索引
        ##### 新图像，用于存储 cross view 并集
        new_instance = torch.zeros_like(instances_current)

        for idx_next in index_list:
            img_next = self._rgbs[idx_next].clone().view(self.H, self.W, 3)
            instances_next = self._labels[idx_next].clone().view(self.H, self.W)
            depth_next = (self._depth_djis[idx_next] * self._depth_scales[idx_next])
            inf_mask = torch.isinf(depth_next)
            depth_next[inf_mask] = depth_next[~inf_mask].max()
            metadata_next = self.metadata_items[self._img_indices[idx_next]]

            ###### 先投影， 这里采用把第二张图（其他图）投回第一张图

            x_grid, y_grid = torch.meshgrid(torch.arange(self.W), torch.arange(self.H))
            x_grid, y_grid = x_grid.T.flatten(), y_grid.T.flatten()
            ## 第二张图先得到点云
            pixel_coordinates = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)
            K1 = metadata_next.intrinsics
            K1 = torch.tensor([[K1[0], 0, K1[2]], [0, K1[1], K1[3]], [0, 0, 1]])
            pt_3d = depth_next[:, None] * (torch.linalg.inv(K1) @ pixel_coordinates[:, :, None].float()).squeeze()
            arr2 = torch.ones((pt_3d.shape[0], 1))
            pt_3d = torch.cat([pt_3d, arr2], dim=-1)
            # pt_3d = pt_3d[valid_depth_mask]
            pt_3d = pt_3d.view(-1, 4)
            E1 = metadata_next.c2w.clone().detach()
            E1 = torch.stack([E1[:, 0], -E1[:, 1], -E1[:, 2], E1[:, 3]], dim=1)
            world_point = torch.mm(torch.cat([E1, torch.tensor([[0, 0, 0, 1]])], dim=0), pt_3d.t()).t()
            world_point = world_point[:, :3] / world_point[:, 3:4]

            ### 投影回第一张图
            E2 = metadata_current.c2w.clone().detach()
            E2 = torch.stack([E2[:, 0], E2[:, 1]*-1, E2[:, 2]*-1, E2[:, 3]], 1)
            w2c = torch.inverse(torch.cat((E2, torch.tensor([[0, 0, 0, 1]])), dim=0))
            points_homogeneous = torch.cat((world_point, torch.ones((world_point.shape[0], 1), dtype=torch.float32)), dim=1)
            pt_3d_trans = torch.mm(w2c, points_homogeneous.t())
            pt_2d_trans = torch.mm(K1, pt_3d_trans[:3])
            pt_2d_trans = pt_2d_trans / pt_2d_trans[2]
            projected_points = pt_2d_trans[:2].t()

            ########### 取得落在图像上的点 mask_x & mask_y ， 并考虑遮挡 mask_z
            threshold= 0.02
            image_width, image_height = self.W, self.H
            mask_x = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < image_width)
            mask_y = (projected_points[:, 1] >= 0) & (projected_points[:, 1] < image_height)
            mask = mask_x & mask_y
            x = projected_points[:, 0].long()
            y = projected_points[:, 1].long()
            x[~mask] = 0
            y[~mask] = 0
            depth_map_current = depth_current[y, x]
            depth_map_current[~mask] = -1e6
            depth_z = pt_3d_trans[2]
            mask_z = depth_z < (depth_map_current + threshold)
            ## 这里拿到了投影的有效 pixel
            mask_xyz = (mask & mask_z)

            x[~mask_xyz] = 0
            y[~mask_xyz] = 0
            
            # 获得了 第二张图投影到第一张图上的 instance label, 除了投影的区域，其他都是0
            # 需要可视化看一下投影的对不对
            project_instance = torch.zeros_like(instances_current)
            project_instance[y[mask_xyz], x[mask_xyz]] = instances_next[y_grid[mask_xyz], x_grid[mask_xyz]]

            ######接下来对每个mask进行cross view 操作
            # project_instance.nonzero().shape[0]> 0.05 * self.H * self.W

            if visualization:
                color_project = torch.zeros_like(img_current)
                color_next = torch.zeros_like(img_current)
                unique_label = torch.unique(instances_next)
                for uni in unique_label:
                    if uni ==0:
                        continue
                    random_color = torch.randint(0, 256, (3,), dtype=torch.uint8)
                    if (instances_next==uni).sum() != 0:
                        color_next[instances_next==uni,:] = random_color
                    if (project_instance==uni).sum() != 0:
                        color_project[project_instance==uni,:] = random_color
                vis_img2 = 0.7 * color_next + 0.3 * img_next
                vis_img3 = 0.7 * color_project + 0.3 * img_current

                if project_instance.nonzero().shape[0]> 0.05 * self.H * self.W:
                    vis_img = np.concatenate([vis_img1.cpu().numpy(), vis_img2.cpu().numpy(), vis_img3.cpu().numpy()], axis=1)
                    cv2.imwrite(f"zyq/1027_crossview_project/test/%06d_%06d.jpg" % (idx, idx_next), vis_img)

            

        if idx == len(self._rgbs) - 1:
            return 'end'
        else:
            print(f"process idx : {idx}")
            return None
    

####选择一个图像，往相机视角方向飞


import click
import os
import numpy as np
import cv2 as cv
from os.path import join as pjoin
from glob import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import torch
from gp_nerf.runner_gpnerf import Runner
from gp_nerf.opts import get_opts_base
from argparse import Namespace
from tqdm import tqdm
import cv2
from tools.unetformer.uavid2rgb import rgb2custom, custom2rgb,remapping

from PIL import Image
from pathlib import Path
import open3d as o3d
import pickle
import math
from dji.process_dji_v8_color import euler2rotation, rad
import xml.etree.ElementTree as ET
from collections import Counter
from pyntcloud import PyntCloud
import pandas as pd
from functools import reduce
from mega_nerf.ray_utils import get_rays, get_ray_directions



torch.cuda.set_device(5)


def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926',required=False, help='')
    parser.add_argument('--exp_name', type=str, default='logs_357/test',required=False, help='experiment name')
    parser.add_argument('--output_path', type=str, default='zyq/1031_crossview_project',required=False, help='experiment name')

    
    return parser.parse_args()


def hello(hparams: Namespace) -> None:
    hparams.ray_altitude_range = [-95, 54]
    hparams.dataset_type='memory_depth_dji_instance_crossview_process'
    hparams.depth_dji_type=='mesh'
    hparams.label_name = 'm2f' # ['m2f', 'merge', 'gt']
    hparams.instance_name = 'instances_mask_0.001' # ['m2f', 'merge', 'gt']
    hparams.sampling_mesh_guidance=True

    runner = Runner(hparams)
    train_items = runner.train_items

    used_files = []
    for ext in ('*.png', '*.jpg'):
        used_files.extend(glob(os.path.join(hparams.dataset_path, 'subset', 'rgbs', ext)))
    used_files.sort()
    process_item = [Path(far_p).stem for far_p in used_files]
    train_items = [train_item for train_item in train_items if (Path(train_item.image_path).stem in process_item)]


    for idx, metadata_item in enumerate(tqdm(train_items, desc="")):
        file_name = Path(metadata_item.image_path).stem
        if file_name not in process_item or metadata_item.is_val: # or int(file_name) != 182:
            continue

        device='cuda'
        overlap_threshold=0.5
        ###NOTE 需要将shuffle调成False, 不打乱，按照顺序处理
        H, W = metadata_item.H, metadata_item.W
        directions = get_ray_directions(metadata_item.W,
                                        metadata_item.H,
                                        metadata_item.intrinsics[0],
                                        metadata_item.intrinsics[1],
                                        metadata_item.intrinsics[2],
                                        metadata_item.intrinsics[3],
                                        False,
                                        device)
        _depth_scales = torch.abs(directions[:, :, 2]).to(device)


        # 拿到当前图像的数据  
        img_current = metadata_item.load_image().view(H, W, 3).to(device)
        instances_current = metadata_item.load_instance().view(H, W).to(device)
        depth_current = (metadata_item.load_depth_dji().view(H, W).to(device) * _depth_scales)

        metadata_current = metadata_item


        # if int(Path(metadata_current.image_path).stem) < 200 and int(Path(metadata_current.image_path).stem) > 300:
            # return None
        
        visualization = True
        # if visualization:
        #     color_current = torch.zeros_like(img_current)
        #     unique_label = torch.unique(instances_current)
        #     for uni in unique_label:
        #         if uni ==0:
        #             continue
        #         random_color = torch.randint(0, 256, (3,), dtype=torch.uint8).to(device)
        #         if (instances_current==uni).sum() != 0:
        #             color_current[instances_current==uni,:] = random_color
        #     vis_img1 = 0.7 * color_current + 0.3 * img_current
        #     Path(f"{hparams.output_path}/test_{overlap_threshold}/mask_vis").mkdir(exist_ok=True, parents=True)

            # cv2.imwrite(f"{hparams.output_path}/test_{overlap_threshold}/mask_vis/%06d.jpg" % (idx), color_current.cpu().numpy())




        index_list= list(range(len(train_items)))
        index_list.remove(idx)  # 从列表中移除已知的索引
        ##### 新图像，用于存储 cross view 并集
        new_instance = torch.zeros_like(instances_current)

        unique_labels,counts = torch.unique(instances_current, return_counts=True)
        sorted_indices = torch.argsort(counts, descending=True)
        sorted_labels = unique_labels[sorted_indices]

        ## 每一个mask进行操作
        for unique_label in unique_labels:
            ####  为每一个mask创建一个list， 存储 需要合并的mask和 overlap分数
            if unique_label==0:
                continue
            merge_unique_label_list = []
            score_list = []

            mask_idx = instances_current == unique_label
            mask_idx_area = mask_idx.sum()
            if mask_idx_area < 0.005 * H * W:
                continue

            # 把投影结果先存储下来，避免重复投影
            project_instances = [] 
            # 先进行投影
            for idx_next in index_list:
                metadata_next = train_items[idx_next]
                img_next = metadata_next.load_image().view(H, W, 3).to(device)
                instances_next = metadata_next.load_instance().view(H, W).to(device)
                depth_next = (metadata_next.load_depth_dji().view(H, W).to(device) * _depth_scales).view(-1)

                inf_mask = torch.isinf(depth_next)
                depth_next[inf_mask] = depth_next[~inf_mask].max()

                ###### 先投影， 这里采用把第二张图（其他图）投回第一张图

                x_grid, y_grid = torch.meshgrid(torch.arange(W), torch.arange(H))
                x_grid, y_grid = x_grid.T.flatten().to(device), y_grid.T.flatten().to(device)
                ## 第二张图先得到点云
                pixel_coordinates = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)
                K1 = metadata_next.intrinsics
                K1 = torch.tensor([[K1[0], 0, K1[2]], [0, K1[1], K1[3]], [0, 0, 1]]).to(device)
                pt_3d = depth_next[:, None] * (torch.linalg.inv(K1) @ pixel_coordinates[:, :, None].float()).squeeze()
                arr2 = torch.ones((pt_3d.shape[0], 1)).to(device)
                pt_3d = torch.cat([pt_3d, arr2], dim=-1)
                # pt_3d = pt_3d[valid_depth_mask]
                pt_3d = pt_3d.view(-1, 4)
                E1 = metadata_next.c2w.clone().detach()
                E1 = torch.stack([E1[:, 0], -E1[:, 1], -E1[:, 2], E1[:, 3]], dim=1).to(device)
                world_point = torch.mm(torch.cat([E1, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0), pt_3d.t()).t()
                world_point = world_point[:, :3] / world_point[:, 3:4]

                ### 投影回第一张图
                E2 = metadata_current.c2w.clone().detach()
                E2 = torch.stack([E2[:, 0], E2[:, 1]*-1, E2[:, 2]*-1, E2[:, 3]], 1).to(device)
                w2c = torch.inverse(torch.cat((E2, torch.tensor([[0, 0, 0, 1]], device=device)), dim=0))
                points_homogeneous = torch.cat((world_point, torch.ones((world_point.shape[0], 1), dtype=torch.float32, device=device)), dim=1)
                pt_3d_trans = torch.mm(w2c, points_homogeneous.t())
                pt_2d_trans = torch.mm(K1, pt_3d_trans[:3])
                pt_2d_trans = pt_2d_trans / pt_2d_trans[2]
                projected_points = pt_2d_trans[:2].t()

                ########### 取得落在图像上的点 mask_x & mask_y ， 并考虑遮挡 mask_z
                threshold= 0.02
                image_width, image_height = W, H
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
                # project_instance.nonzero().shape[0]> 0.05 * H * W
                project_instances.append(project_instance)
                
        
            ## 这里改为800张图像的投影结束后， 进行overlap计算  
            for idx_next_i, idx_next in enumerate(index_list):
                project_instance=project_instances[idx_next_i]
                label_in_mask = project_instance[mask_idx]
                uni_label_in_mask, count_label_in_mask = torch.unique(label_in_mask, return_counts=True)
                for uni_2 in uni_label_in_mask:
                    if uni_2 == 0:
                        continue
                    mask_2 = project_instance == uni_2
                    mask_area_2 = mask_2.sum()
                    mask_area_overlap = (mask_idx * mask_2).sum()
                    # 存储符合条件的并集mask 和 对应要融合区域大小的score
                    if (mask_area_overlap / mask_area_2) > overlap_threshold or (mask_area_overlap / mask_idx_area) > overlap_threshold:
                        if mask_area_2 < 0.001 * H * W or (instances_next==uni_2).sum() < 0.001 * H * W:
                            continue
                        merge_unique_label_list.append(mask_2)
                        score_list.append(mask_area_2)

                        # if visualization:
                        if False:
                            color_project = torch.zeros_like(img_current)
                            color_next = torch.zeros_like(img_current)
                            random_color = torch.randint(0, 256, (3,), dtype=torch.uint8).to(device)
                            if (instances_next==uni_2).sum() != 0:
                                color_next[instances_next==uni_2,:] = random_color
                            if (project_instance==uni_2).sum() != 0:
                                color_project[project_instance==uni_2,:] = random_color

                            vis_img2 = 0.7 * color_next + 0.3 * img_next
                            vis_img3 = 0.7 * color_project + 0.3 * img_current

                            # if project_instance.nonzero().shape[0]> 0.05 * H * W:
                            vis_img = np.concatenate([vis_img1.cpu().numpy(), vis_img2.cpu().numpy(), vis_img3.cpu().numpy()], axis=1)
                            Path(f"{hparams.output_path}/test_{overlap_threshold}/each").mkdir(exist_ok=True, parents=True)
                            cv2.imwrite(f"{hparams.output_path}/test_{overlap_threshold}/each/%06d_label%06d_%06d.jpg" % (int(Path(metadata_current.label_path).stem), unique_label, int(Path(metadata_next.label_path).stem)), vis_img)
                    
                            

            if merge_unique_label_list != []:
                sorted_data = sorted(zip(merge_unique_label_list, score_list), key=lambda x: x[1], reverse=False)

                merge_unique_label_list, score_list = zip(*sorted_data)
                merge_unique_label_list = list(merge_unique_label_list)
                merge_unique_label_list.append(mask_idx)

                ###得到一个mask 在所有图像上需要融合的区域后， 对所有的mask求并集
                # iiiii=1
                # for iii_mask in merge_unique_label_list:
                #     color_result = torch.zeros_like(img_current)
                #     random_color = torch.randint(0, 256, (3,), dtype=torch.uint8)

                #     color_result[iii_mask]=random_color
                #     cv2.imwrite(f"{hparams.output_path}/test_{overlap_threshold}/results/%06d_vis.jpg" % (iiiii), color_result.cpu().numpy())
                #     iiiii += 1

                union_mask = reduce(torch.logical_or, merge_unique_label_list)
                new_instance[union_mask] = unique_label





        if new_instance.sum() != 0:
            color_result = torch.zeros_like(img_current)

            color_current = torch.zeros_like(img_current)
            unique_label_results = torch.unique(instances_current)
            for uni in unique_label_results:
                if uni ==0:
                    continue
                random_color = torch.randint(0, 256, (3,), dtype=torch.uint8).to(device)
                color_current[instances_current==uni,:] = random_color
                color_result[new_instance==uni,:] = random_color
                
            vis_img1 = 0.7 * color_current + 0.3 * img_current
            vis_img4 = 0.7 * color_result + 0.3 * img_current

            
            vis_img5 = np.concatenate([vis_img1.cpu().numpy(), vis_img4.cpu().numpy()], axis=1)

            Path(f"{hparams.output_path}/test_{overlap_threshold}/results").mkdir(exist_ok=True, parents=True)
            cv2.imwrite(f"{hparams.output_path}/test_{overlap_threshold}/results/%06d_results_%06d.jpg" % (int(Path(metadata_current.label_path).stem), unique_label), vis_img5)
            Path(f"{hparams.output_path}/test_{overlap_threshold}/crossview").mkdir(exist_ok=True, parents=True)
            Image.fromarray(new_instance.cpu().numpy().astype(np.uint8)).save(f"{hparams.output_path}/test_{overlap_threshold}/crossview/{Path(metadata_current.label_path).stem}.png")

            

    print('done')


if __name__ == '__main__':
    hello(_get_train_opts())

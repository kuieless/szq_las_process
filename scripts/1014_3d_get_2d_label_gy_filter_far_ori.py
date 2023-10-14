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




def calculate_entropy(labels):
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    total_labels = len(labels)

    probabilities = label_counts / total_labels
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def custom2rgb_1(mask):
    N= mask.shape[0]
    mask_rgb = np.zeros(shape=(N, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]             # cluster       black
    
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 0, 0]           # building      red
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [192, 192, 192]       # road        grey  
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [192, 0, 192]         # car           light violet
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 128, 0]           # tree          green
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [128, 128, 0]         # vegetation    dark green
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 255, 0]         # human         yellow
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [135, 206, 250]       # sky           light blue
    mask_rgb[np.all(mask_convert == 8, axis=0)] = [0, 0, 128]           # water         blue

    mask_rgb[np.all(mask_convert == 9, axis=0)] = [252,230,201]          # ground       egg
    mask_rgb[np.all(mask_convert == 10, axis=0)] = [128, 64, 128]        # mountain     dark violet

    # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb



def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926',required=False, help='')
    parser.add_argument('--exp_name', type=str, default='logs_357/test',required=False, help='experiment name')
    parser.add_argument('--far_paths', type=str, default='logs_dji/1003_yingrenshi_density_depth_hash22_semantic/18/eval_200000_far0.5/yingrenshi_panoptic_car_augment_far_0.5/labels_m2f',required=False, help='experiment name')
    
    parser.add_argument('--output_path', type=str, default='zyq/1014_3d_get2dlabel_test',required=False, help='experiment name')
    # parser.add_argument('--output_path', type=str, default='zyq/1010_3d_get2dlabel_gt_test',required=False, help='experiment name')

    
    return parser.parse_args()


def hello(hparams: Namespace) -> None:


    far_paths = hparams.far_paths
    output_path = hparams.output_path
    if not os.path.exists(output_path):
        Path(output_path).mkdir(parents=True)
    if not os.path.exists(os.path.join(output_path, 'project_far_to_ori')):
        Path(os.path.join(output_path, 'project_far_to_ori')).mkdir(parents=True)
    if not os.path.exists(os.path.join(output_path, 'vis')):
        Path(os.path.join(output_path, 'vis')).mkdir(parents=True)
    if not os.path.exists(os.path.join(output_path, 'alpha')):
        Path(os.path.join(output_path, 'alpha')).mkdir(parents=True)


    hparams.ray_altitude_range = [-95, 54]
    hparams.dataset_type='memory_depth_dji'
    device = 'cpu'
    hparams.label_name = 'm2f' # ['m2f', 'merge', 'gt']
    runner = Runner(hparams)
    train_items = runner.train_items



    used_files = []
    for ext in ('*.png', '*.jpg'):
        used_files.extend(glob(os.path.join(far_paths, ext)))
    used_files.sort()
    process_item = [Path(far_p).stem for far_p in used_files]
    
    for metadata_item in tqdm(train_items, desc="Processing project 2d to 3d point"):
        file_name = Path(metadata_item.image_path).stem
        if file_name not in process_item:
            continue

        # 读取ori的标签文件
        ori_m2f = metadata_item.load_label()

        # 读取深度
        depth_map = metadata_item.load_depth_dji().squeeze(-1)
        H, W = depth_map.shape

        ## 1. 用2d和depth转换成点云
        x_grid, y_grid = torch.meshgrid(torch.arange(W), torch.arange(H))
        # x_grid, y_grid = torch.meshgrid(torch.arange(H), torch.arange(W))
        pixel_coordinates = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)
        K1 = metadata_item.intrinsics
        K1 = torch.tensor([[K1[0], 0, K1[2]], [0, K1[1], K1[3]], [0, 0, 1]])
        pt_3d = depth_map[:, :, None] * (torch.linalg.inv(K1) @ pixel_coordinates[:, :, :, None].float()).squeeze()
        arr2 = torch.ones((pt_3d.shape[0], pt_3d.shape[1], 1))
        pt_3d = torch.cat([pt_3d, arr2], dim=-1)
        # pt_3d = pt_3d[valid_depth_mask]
        pt_3d = pt_3d.view(-1, 4)
        E1 = torch.tensor(metadata_item.c2w)
        E1 = torch.stack([E1[:, 0], -E1[:, 1], -E1[:, 2], E1[:, 3]], dim=1)
        world_point = torch.mm(torch.cat([E1, torch.tensor([[0, 0, 0, 1]])], dim=0), pt_3d.t()).t()
        world_point = world_point[:, :3] / world_point[:, 3:4]



        # 点云投影到far
        metadata_far = torch.load(os.path.join(hparams.dataset_path, 'render_far', 'metadata', file_name+'.pt'), map_location='cpu')


        camera_rotation = metadata_far['c2w'][:3,:3].to(device)
        camera_position = metadata_far['c2w'][:3, 3].to(device)
        camera_matrix = torch.tensor([[metadata_item.intrinsics[0], 0, metadata_item.intrinsics[2]],
                              [0, metadata_item.intrinsics[1], metadata_item.intrinsics[3]],
                              [0, 0, 1]]).to(device)

        
        # NOTE: 2. 自己写，正确  
        E2 = torch.hstack((camera_rotation, camera_position.unsqueeze(-1)))
        E2 = torch.stack([E2[:, 0], E2[:, 1]*-1, E2[:, 2]*-1, E2[:, 3]], 1).to(device)
        w2c = torch.inverse(torch.cat((E2, torch.tensor([[0, 0, 0, 1]]).to(device)), dim=0))
        points_homogeneous = torch.cat((world_point, torch.ones((world_point.shape[0], 1), dtype=torch.float32).to(device)), dim=1)
        pt_3d_trans = torch.mm(w2c, points_homogeneous.t())
        
        pt_2d_trans = torch.mm(camera_matrix, pt_3d_trans[:3])
        pt_2d_trans = pt_2d_trans / pt_2d_trans[2]
        projected_points = pt_2d_trans[:2].t()
        
        x = projected_points[:, 0]
        y = projected_points[:, 1]

        nan_indices = torch.isnan(x)
        prev_indices = torch.arange(len(x) - 1)
        next_indices = torch.arange(1, len(x))
        x[torch.where(nan_indices)] = (x[prev_indices][nan_indices[1:]] + x[next_indices][nan_indices[:-1]]) / 2

        nan_indices = torch.isnan(y)
        prev_indices = torch.arange(len(y) - 1)
        next_indices = torch.arange(1, len(y))
        y[torch.where(nan_indices)] = (y[prev_indices][nan_indices[1:]] + y[next_indices][nan_indices[:-1]]) / 2


        x = x.long()
        y = y.long()


        # 读取far的标签文件
        far_m2f = Image.open(os.path.join(far_paths, file_name+'.png'))    #.convert('RGB')
        far_m2f = torch.ByteTensor(np.asarray(far_m2f))

        project_m2f = far_m2f[y, x].view(H,W)
        
        Image.fromarray(project_m2f.numpy().astype(np.uint16)).save(os.path.join(output_path, 'project_far_to_ori', f"{file_name}.png"))
        

        color_label = custom2rgb(project_m2f.numpy())
        color_label = color_label.reshape(H, W,3)
        Image.fromarray(color_label.astype(np.uint8)).save(os.path.join(output_path, 'vis',f"{file_name}.jpg"))

        img = metadata_item.load_image()

        merge = 0.7 * img.numpy() + 0.3 * color_label
        Image.fromarray(merge.astype(np.uint8)).save(os.path.join(output_path, 'alpha',f"{file_name}.jpg"))
        

    print('done')


if __name__ == '__main__':
    #meshMask = torch.zeros(912, 1368, 1)
    #projected_points = torch.rand(2222058, 2) * 500

    #x = projected_points[:, 0].long()
    #y = projected_points[:, 1].long()
    #mesh_depths = meshMask[x, y]
    #print(mesh_depths.shape)
    #import pdb; pdb.set_trace()

    hello(_get_train_opts())

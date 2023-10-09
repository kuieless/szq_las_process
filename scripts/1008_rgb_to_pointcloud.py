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
from mega_nerf.ray_utils import get_rays, get_ray_directions
from tools.unetformer.uavid2rgb import remapping
from tools.unetformer.uavid2rgb import custom2rgb
import open3d as o3d
import struct
import pandas as pd

from pyntcloud import PyntCloud

def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926',required=False, help='')
    parser.add_argument('--exp_name', type=str, default='logs_357/test',required=False, help='experiment name')
    
    return parser.parse_args()




def hello(hparams: Namespace) -> None:
    hparams.ray_altitude_range = [-95, 54]
    hparams.dataset_type='memory_depth_dji'



    runner = Runner(hparams)
    train_items = runner.train_items


    split_list = []
    world_points = []
    for metadata_item in tqdm(train_items):
        gt_label = metadata_item.load_gt()
        gt_label = remapping(gt_label)
        non_zero_mask = (gt_label != 0)
        gt_label = custom2rgb(gt_label.cpu().numpy())


        depth_map = metadata_item.load_depth_dji().squeeze(-1)
        valid_depth_mask = ~torch.isinf(depth_map)
        valid_depth_mask = valid_depth_mask * non_zero_mask
        # gt_depths[valid_depth_mask]

        H, W, = depth_map.shape
        
        directions = get_ray_directions(W,
                                        H,
                                        metadata_item.intrinsics[0],
                                        metadata_item.intrinsics[1],
                                        metadata_item.intrinsics[2],
                                        metadata_item.intrinsics[3],
                                        False,
                                        'cpu')
        depth_scale = torch.abs(directions[:, :, 2]) # z-axis's values
        # depth_map = (depth_map * depth_scale).numpy()
        depth_map= depth_map.numpy()
        x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))

        # x, y = int(W * 1/2), int(H * 1/2)
        # depth = depth_map[y, x]   #* 0.5
        # 将像素坐标转换为齐次坐标
        pixel_coordinates = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)
        # pixel_coordinates = pixel_coordinates.reshape(-1,3)
        # depth = depth_map  #.flatten()

        K1 = metadata_item.intrinsics
        K1 = np.array([[K1[0], 0, K1[2]],[0, K1[1], K1[3]],[0,0,1]])

        # pt_3d = np.dot(np.linalg.inv(K1), np.array([x, y, 1])) * depth

        # pt_3d = np.dot(np.linalg.inv(K1), pixel_coordinates.T).T * depth[:,np.newaxis]

        pt_3d = depth_map[:, :, np.newaxis] * (np.linalg.inv(K1) @ pixel_coordinates[:, :, :, np.newaxis]).squeeze()


        arr2 = np.ones((pt_3d.shape[0], pt_3d.shape[1], 1))
        # pt_3d = np.hstack((pt_3d, arr2))
        pt_3d = np.concatenate([pt_3d, arr2], axis=-1)
        pt_3d = pt_3d[valid_depth_mask]


        E1 = np.array(metadata_item.c2w)
        E1 = np.stack([E1[:, 0], E1[:, 1]*-1, E1[:, 2]*-1, E1[:, 3]], 1)
        world_point = np.dot(np.concatenate((E1, [[0,0,0,1]]), 0), pt_3d.T).T
        label_rgb = gt_label[valid_depth_mask]
        world_point[:,:3] = world_point[:,:3] / world_point[:,3:4]

        pc_rgb = np.hstack((world_point[:,:3], label_rgb))

        world_points.append(pc_rgb)

    combined_array = np.concatenate(world_points, axis=0)

    print(combined_array.shape)
    combined_array=combined_array[::10]

    cloud = PyntCloud(pd.DataFrame(
        # same arguments that you are passing to visualize_pcl
        data=np.hstack((combined_array[:, :3], np.uint8(combined_array[:, 3:6]))),
        columns=["x", "y", "z", "red", "green", "blue"]))

    cloud.to_file("point_cloud.ply")

    print('done')




if __name__ == '__main__':
    hello(_get_train_opts())

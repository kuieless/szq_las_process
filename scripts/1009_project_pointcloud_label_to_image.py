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
from tools.unetformer.uavid2rgb import rgb2custom, custom2rgb
from PIL import Image
from pathlib import Path
import open3d as o3d


def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926',required=False, help='')
    parser.add_argument('--exp_name', type=str, default='logs_357/test',required=False, help='experiment name')
    parser.add_argument('--output_path', type=str, default='zyq/pc2label',required=False, help='experiment name')

    
    return parser.parse_args()


def hello(hparams: Namespace) -> None:
    hparams.ray_altitude_range = [-95, 54]
    hparams.dataset_type='memory_depth_dji'


    threshold=0.015

    point_cloud = o3d.io.read_point_cloud("zyq/2d-3d-2d_yingrenshi_m2f/point_cloud_full.ply")

    points_nerf = np.asarray(point_cloud.points)
    points_color = np.asarray(point_cloud.colors)*255


    print(points_nerf.shape)
    print(points_color.shape)


    runner = Runner(hparams)
    val_items = runner.val_items


    split_list=[]
    for metadata_item in tqdm(val_items):
        H, W = metadata_item.H, metadata_item.W
        camera_rotation = metadata_item.c2w[:3,:3]
        camera_position = metadata_item.c2w[:3, 3]
        camera_matrix = np.array([[metadata_item.intrinsics[0], 0, metadata_item.intrinsics[2]],
                                [0, metadata_item.intrinsics[1], metadata_item.intrinsics[3]],
                                [0, 0, 1]])
        # NOTE: 2. 自己写，正确
        E2 = np.hstack((camera_rotation, camera_position[:, np.newaxis]))
        E2 = np.stack([E2[:, 0], E2[:, 1]*-1, E2[:, 2]*-1, E2[:, 3]], 1)
        w2c = np.linalg.inv(np.concatenate((E2, [[0,0,0,1]]), 0))
        points_homogeneous = np.hstack((points_nerf, np.ones((points_nerf.shape[0], 1))))
        pt_3d_trans = np.dot(w2c, points_homogeneous.T)

        pt_2d_trans = np.dot(camera_matrix, pt_3d_trans[:3]) 
        pt_2d_trans = pt_2d_trans / pt_2d_trans[2]
        projected_points = (pt_2d_trans)[:2, :]




        # 创建空白图像
        large_int = 1e6
        image_width, image_height = int(W), int(H)

        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        image_expand = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        image_expand_depth = np.zeros((image_height, image_width, 3), dtype=np.uint8)


        large_int = 1e6
        depth_map = large_int * np.ones((image_height, image_width, 1), dtype=np.uint8)
        depth_map_expand = large_int * np.ones((image_height, image_width, 1), dtype=np.uint8)

        # 获得落在图像上的点
        mask_x = np.logical_and(projected_points[0, :] >= 0, projected_points[0, :] < image_width)
        mask_y = np.logical_and(projected_points[1, :] >= 0, projected_points[1, :] < image_height)
        mask = np.logical_and(mask_x, mask_y)



        depth_z = pt_3d_trans[2, mask]

        projected_points_mask = projected_points[:, mask]
        depth_z_mask = depth_z[mask]



        step = 3
        points_color_mask = points_color[mask]
        expand=True
        meshMask = metadata_item.load_depth_dji().cpu().numpy()

        for x, y, depth, color in tqdm(zip(projected_points_mask[0], projected_points_mask[1], depth_z, points_color_mask)):
            x, y = int(x), int(y)
            if depth < depth_map[y, x]:
                depth_map[y, x] = depth
                image[y, x] = color[::-1]
                # cv2.circle(image_expand, (x, y), 2, (float(color[2]), float(color[1]), float(color[0])), -1)
            if expand and depth < depth_map_expand[y, x]:
                depth_map_expand[max(0, y - step):min(image_height, y + step), max(0, x - step):min(image_width, x + step)] = depth
        
        image = image[:, :, ::-1]
        image_label = rgb2custom(image)



        output_path = hparams.output_path
        if not os.path.exists(output_path):
            Path(output_path).mkdir(parents=True)
            Path(os.path.join(output_path, 'vis')).mkdir(parents=True)
            Path(os.path.join(output_path, 'val_vis')).mkdir(parents=True)
            Path(os.path.join(output_path, 'labels_pc')).mkdir(parents=True)

        Image.fromarray(image_label.astype(np.uint16)).save(os.path.join(output_path, 'labels_pc', f"{metadata_item.image_path.stem}.png"))
        Image.fromarray(image.astype(np.uint8)).save(os.path.join(output_path, 'vis', f"{metadata_item.image_path.stem}.png"))

        gt_label_rgb = custom2rgb(image_label)
        Image.fromarray(gt_label_rgb.astype(np.uint8)).save(os.path.join(output_path, 'val_vis', f"{metadata_item.image_path.stem}.png"))


    print("done")




if __name__ == '__main__':
    hello(_get_train_opts())

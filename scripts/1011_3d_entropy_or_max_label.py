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
from tools.unetformer.uavid2rgb import rgb2custom
from PIL import Image
from pathlib import Path
import open3d as o3d
import pickle
from pyntcloud import PyntCloud
import pandas as pd
from collections import Counter
from tools.unetformer.uavid2rgb import remapping
from tools.unetformer.uavid2rgb import custom2rgb

def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--output_path', type=str, default='zyq/1010_3d_get2dlabel',required=False, help='experiment name')

    
    return parser.parse_args()


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

        

def hello(hparams: Namespace) -> None:
    output_path = hparams.output_path
    
    
    # # # ply 文件
    point_cloud = o3d.io.read_point_cloud("zyq/2d-3d-2d_yingrenshi_m2f/point_cloud_50.ply")
    points_nerf = np.asarray(point_cloud.points)

    with open('/data/yuqi/code/GP-NeRF-semantic/zyq/1010_3d_get2dlabel/point_label_list_gy.pkl', 'rb') as file:
        # 使用 pickle.load() 读取数据
        loaded_data = pickle.load(file)

    ## 1. entropy

    #######################  计算entropy
    

    # entropies = [calculate_entropy(point) for point in tqdm(loaded_data)]
    # entropies_intensity = np.array(entropies)
    # np.save(f"{output_path}/entropies_intensity.npy", entropies_intensity)


    ######## 保存
    entropies_intensity = np.load(f"{output_path}/entropies_intensity.npy")

    min_entropy = min(entropies_intensity)
    max_entropy = max(entropies_intensity)
    normalized_intensities = (entropies_intensity - min_entropy) / (max_entropy - min_entropy) * 255


    cloud = PyntCloud(pd.DataFrame(
        # same arguments that you are passing to visualize_pcl
        data=np.hstack((points_nerf[:, :3], normalized_intensities[:, np.newaxis])),
        columns=["x", "y", "z", "intensity"]))
    cloud.to_file(f"{output_path}/entropy_intensity_pc.ply")

    ### 2. 投票峰值


    # most_common_labels = []
    # for point in loaded_data:
    #     if not point:
    #         # 处理空列表的情况
    #         most_common_labels.append(None)
    #     else:
    #         labels_array = np.array(point)
    #         most_common_label = np.argmax(np.bincount(labels_array))
    #         most_common_labels.append(most_common_label)


    most_common_labels = []
    for point in loaded_data:
        if not point:
            # 处理空列表的情况
            most_common_labels.append(-1)
        else:
            most_common_labels.append(Counter(point).most_common(1)[0][0])
    
    most_common_labels = np.array(most_common_labels)
    max_label = remapping(most_common_labels)


    
    max_label_color = custom2rgb_1(max_label)



    cloud = PyntCloud(pd.DataFrame(
        # same arguments that you are passing to visualize_pcl
        data=np.hstack((points_nerf[:, :3], np.uint8(max_label_color))),
        columns=["x", "y", "z", "red", "green", "blue"]))
    cloud.to_file(f"{output_path}/most_label_pc.ply")


    print('done')

if __name__ == '__main__':
    hello(_get_train_opts())

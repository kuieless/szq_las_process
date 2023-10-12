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

import math

from dji.process_dji_v8_color import euler2rotation, rad
import xml.etree.ElementTree as ET

def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926',required=False, help='')

    parser.add_argument('--output_path', type=str, default='zyq/1010_3d_get2dlabel_gt_1924_fliter',required=False, help='experiment name')
    parser.add_argument('--metaXml_path', default='/data/yuqi/Datasets/DJI/origin/Yingrenshi_20230926_origin/terra_point_ply/metadata.xml', type=str)
    
    return parser.parse_args()


def calculate_entropy(labels):
    # labels = np.array(labels)
    # filtered_labels = labels[labels != 0]
    # unique_labels, label_counts = np.unique(filtered_labels, return_counts=True)
    # total_labels = len(filtered_labels)
    # total_labels = int(filtered_labels.size)


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
    
    
    # #1. txt文件
    points_nerf = np.genfromtxt('/data/yuqi/Datasets/DJI/origin/Yingrenshi_fine-registered.txt', usecols=(0, 1, 2))
    print(points_nerf.shape)
    root = ET.parse(hparams.metaXml_path).getroot()
    translation = np.array(root.find('SRSOrigin').text.split(',')).astype(np.float) 
    coordinate_info = torch.load(hparams.dataset_path + '/coordinates.pt')
    origin_drb = coordinate_info['origin_drb'].numpy()
    pose_scale_factor = coordinate_info['pose_scale_factor']
    ZYQ = torch.DoubleTensor([[0, 0, -1],
                            [0, 1, 0],
                            [1, 0, 0]])
    ZYQ_1 = torch.DoubleTensor([[1, 0, 0],
                            [0, math.cos(rad(135)), math.sin(rad(135))],
                            [0, -math.sin(rad(135)), math.cos(rad(135))]])      
    points_nerf = np.array(points_nerf)
    points_nerf += translation
    points_nerf = ZYQ.numpy() @ points_nerf.T
    points_nerf = (ZYQ_1.numpy() @ points_nerf).T
    points_nerf = (points_nerf - origin_drb) / pose_scale_factor

    # if not os.path.exists(f"{output_path}/ori_color_nerfcoor.ply"):   
    #     points_color = np.genfromtxt('/data/yuqi/Datasets/DJI/origin/Yingrenshi_fine-registered.txt', usecols=(2, 3, 4, 5))
    #     points_color = np.array(points_color)

    #     cloud = PyntCloud(pd.DataFrame(
    #         # same arguments that you are passing to visualize_pcl
    #         data=np.hstack((points_nerf[:, :3], np.uint8(points_color))),
    #         columns=["x", "y", "z", "red", "green", "blue", "label"]))
    #     cloud.to_file(f"{output_path}/ori_color_nerfcoor.ply")

    # # # ply 文件
    # point_cloud = o3d.io.read_point_cloud("zyq/2d-3d-2d_yingrenshi_m2f/point_cloud_50.ply")
    # points_nerf = np.asarray(point_cloud.points)



    with open(f'{output_path}/point_label_list_gy.pkl', 'rb') as file:
        # 使用 pickle.load() 读取数据
        loaded_data = pickle.load(file)

    loaded_data = [[label for label in point if label != 0] for point in loaded_data]

    ###3 . label_num
    print('calculate label_num')
    label_counts = np.array([len(point) for point in loaded_data])
    print(f"label_counts max :{max(label_counts)}, min :{(min(label_counts))}")



    ### 1. entropy
    print('calculate entropy')
    entropies = [calculate_entropy(point) for point in tqdm(loaded_data)]
    entropies_intensity = np.array(entropies)
    np.save(f"{output_path}/entropies_fliter.npy", entropies_intensity)
    ######## 保存
    # entropies_intensity = np.load(f"{output_path}/entropies_fliter.npy")
    min_entropy = min(entropies_intensity)
    max_entropy = max(entropies_intensity)
    normalized_intensities = (entropies_intensity - min_entropy) / (max_entropy - min_entropy) * 255



    ### 2. 投票峰值
    print('calculate most_common_labels')
    most_common_labels = []
    for point in loaded_data:
        if not point:
            # 处理空列表的情况
            most_common_labels.append(-1)
        else:
            most_common_labels.append(Counter(point).most_common(1)[0][0])

            # # 将 point 转换为 NumPy 数组
            # point_array = np.array(point)
            # # 找出不等于0的标签
            # non_zero_labels = point_array[point_array != 0]
            # if non_zero_labels.size > 0:
            #     # 计算标签峰值
            #     most_common = Counter(non_zero_labels).most_common(1)
            #     most_common_labels.append(most_common[0][0])
            # else:
            #     most_common_labels.append(-1)

    
    most_common_labels = np.array(most_common_labels)
    max_label = remapping(most_common_labels)
    max_label_color = custom2rgb_1(max_label)


    cloud = PyntCloud(pd.DataFrame(
        data=np.hstack((points_nerf[:, :3], np.uint8(max_label_color), normalized_intensities[:, np.newaxis], label_counts[:, np.newaxis])),
        columns=["x", "y", "z", "red", "green", "blue", 'entropy', 'label_num']))
    cloud.to_file(f"{output_path}/results.ply")

    # cloud = PyntCloud(pd.DataFrame(
    #     # same arguments that you are passing to visualize_pcl
    #     data=np.hstack((points_nerf[1:, :3], np.uint8(max_label_color[1:]), normalized_intensities[1:, np.newaxis], label_counts[1:, np.newaxis])),
    #     columns=["x", "y", "z", "red", "green", "blue", 'entropy', 'label_num']))
    # cloud.to_file(f"{output_path}/results.ply")


    print('done')

if __name__ == '__main__':
    hello(_get_train_opts())

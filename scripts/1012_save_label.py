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
    parser.add_argument('--output_path', type=str, default='zyq/1010_3d_get2dlabel_gt_1924_fliter',required=False, help='experiment name')
    parser.add_argument('--metaXml_path', default='/data/yuqi/Datasets/DJI/origin/Yingrenshi_20230926_origin/terra_point_ply/metadata.xml', type=str)
    
    return parser.parse_args()



def hello(hparams: Namespace) -> None:
    output_path = hparams.output_path
    
    
    # #1. txt文件
    points_nerf = np.genfromtxt('/data/jxchen/dataset/dji/Yingrenshi/dji_labeled_segmented_ply.txt', usecols=(0, 1, 2, 3, 4, 5))
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
    points_nerf_coor = points_nerf[:,:3]
    points_nerf_coor += translation
    points_nerf_coor = ZYQ.numpy() @ points_nerf_coor.T
    points_nerf_coor = (ZYQ_1.numpy() @ points_nerf_coor).T
    points_nerf_coor = (points_nerf_coor - origin_drb) / pose_scale_factor




    data = np.hstack((points_nerf_coor[:, :3],points_nerf[:, 3:6]))

    np.savetxt("/data/yuqi/Datasets/DJI/origin/Yingrenshi_dji_remappingcolor_nerfcoor.txt", data, fmt=('%f', '%f', '%f') + ('%d',) * 3, delimiter=' ')



    # cloud = PyntCloud(pd.DataFrame(
    #     # same arguments that you are passing to visualize_pcl
    #     data=np.hstack((points_nerf_coor[:, :3], np.uint8(points_nerf[:, 3:]))),
    #     columns=["x", "y", "z", "red", "green", "blue"]))
    # cloud.to_file(f"/data/yuqi/Datasets/DJI/origin/Yingrenshi_dji_remappingcolor_nerfcoor.txt")

    print('done')

if __name__ == '__main__':
    hello(_get_train_opts())

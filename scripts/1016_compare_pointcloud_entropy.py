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
from torch.nn.functional import interpolate
from mega_nerf.ray_utils import get_ray_directions


torch.cuda.set_device(6)


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
    parser.add_argument('--ply1', type=str, default='/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_ori_nerf/3d_get2dlabel/results.ply',required=False, help='')
    parser.add_argument('--ply2', type=str, default='/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.5/3d_get2dlabel/results.ply',required=False, help='experiment name')
    
    parser.add_argument('--output_path', type=str, default='zyq/test',required=False, help='experiment name')
    
    return parser.parse_args()


def hello(hparams: Namespace) -> None:
    cloud1 = PyntCloud.from_file(hparams.ply1)
    cloud2 = PyntCloud.from_file(hparams.ply2)    

    pts1 = torch.from_numpy(np.array(cloud1.points))
    pts2 = torch.from_numpy(np.array(cloud2.points))

    pts_result = torch.zeros((pts1.shape[0],8))
    pts_result[:,:3] = pts1[:,:3]
    
    pts_result[:,3:6] = torch.where(pts1[:,6:7] < pts2[:,6:7], pts1[:,3:6], pts2[:,3:6])
    pts_result[:,6] = torch.where(pts1[:,6] < pts2[:,6], pts1[:,6], pts2[:,6])
    pts_result[:,7] = torch.where(pts1[:,6] < pts2[:,6], pts1[:,7], pts2[:,7])



    cloud = PyntCloud(pd.DataFrame(
        data=np.hstack((pts_result[:, :3], np.uint8(pts_result[:, 3:6]), pts_result[:, 6:7], pts_result[:, 7:8])),
        columns=["x", "y", "z", "red", "green", "blue", 'entropy', 'label_num']))
    cloud.to_file(f"results.ply")

    
    print('done')


if __name__ == '__main__':

    hello(_get_train_opts())

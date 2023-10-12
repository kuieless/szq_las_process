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


    ori_pc = np.genfromtxt('/data/yuqi/Datasets/DJI/origin/Yingrenshi_whole_scene.txt')
    ori_pc[:,0:3] = points_nerf[:,0:3]

    np.savetxt('/data/yuqi/Datasets/DJI/origin/Yingrenshi_whole_scene_nerfcoor.txt', ori_pc, fmt=('%f', '%f', '%f') + ('%d',) * 6, delimiter=' ')

        

    print('done')

if __name__ == '__main__':
    hello(_get_train_opts())

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

import configargparse
from tools.unetformer.uavid2rgb import uavid2rgb, custom2rgb, remapping, remapping_remove_ground


torch.cuda.set_device(4)


def _get_train_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds',required=False, help='')
    parser.add_argument('--output_path', type=str, default='b2_gt_label/change',required=False, help='experiment name')
    return parser.parse_args()

def label_to_color(label_array):
    unique_labels = np.unique(label_array)
    color_image = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)

    for i in unique_labels:
        if i == 0:
            continue
        color = np.random.randint(0, 256, (3,), dtype=np.uint8)
        color_image[label_array == i] = color

    return color_image

def hello(hparams: Namespace) -> None:
    rgb_path = os.path.join(hparams.dataset_path, 'val', 'rgbs')    

    semantic_paths = []
    for ext in ('*.png', '*.jpg'):
        semantic_paths.extend(glob(os.path.join('b2_gt_label/labels_gt1', ext)))
    semantic_paths.sort()


    instance_paths = []
    for ext in ('*.png', '*.jpg'):
        instance_paths.extend(glob(os.path.join('b2_gt_label/instances_gt1', ext)))
    instance_paths.sort()


    H, W = 1024,1536
    for idx, semantic in enumerate(tqdm(semantic_paths)):
        file_name = Path(semantic).stem
        instance = instance_paths[idx]

        rgb = Image.open(os.path.join(rgb_path,f'{file_name}.jpg')).convert('RGB')
        size = rgb.size
        if size[0] != W or size[1] != H:
            rgb = rgb.resize((W, H), Image.LANCZOS)
        rgb = np.asarray(rgb)


        semantic_label = np.array(Image.open(semantic))
        instance_label = np.array(Image.open(instance))
        semantic_label_ori = semantic_label.copy()
        instance_label_ori = instance_label.copy()


        building_mask = semantic_label==1
        instance_label[~building_mask] = 0 


        to_replac_road = [531,553,556, 557,550,551,693,696, 360,361, 374, 388,389,394,559, 551,556,537,541,547,558,696,550,559,555]
    
        to_replac_tree = [397,537]

        remapping_road_mask = np.isin(instance_label, to_replac_road)
        remapping_tree_mask = np.isin(instance_label, to_replac_tree)

        instance_label[remapping_road_mask] = 0
        semantic_label[remapping_road_mask] = 2

        instance_label[remapping_tree_mask] = 0
        semantic_label[remapping_tree_mask] = 4       

        instance_label[instance_label==524] = 525  # 三个车棚
        instance_label[instance_label==526] = 525




        Path(f"{hparams.output_path}/instances_gt").mkdir(exist_ok=True, parents=True)
        Image.fromarray(instance_label.astype(np.uint16)).save(f"{hparams.output_path}/instances_gt/{file_name}.png")


        Path(f"{hparams.output_path}/labels_gt").mkdir(exist_ok=True, parents=True)
        Image.fromarray(semantic_label.astype(np.uint16)).save(f"{hparams.output_path}/labels_gt/{file_name}.png")


        alpha=0.65
        semantic_ori_viz = custom2rgb(semantic_label_ori)*alpha+rgb*(1-alpha)
        semantic_viz = custom2rgb(semantic_label)*alpha+rgb*(1-alpha)


        instance_ori_viz = label_to_color(instance_label_ori)*alpha+rgb*(1-alpha)
        instance_viz = label_to_color(instance_label)*alpha+rgb*(1-alpha)

        viz1 = np.concatenate([semantic_ori_viz, instance_ori_viz],1)
        viz2 = np.concatenate([semantic_viz, instance_viz],1)
        viz3 = np.concatenate([viz1, viz2],0)

        Path(f"{hparams.output_path}/viz").mkdir(exist_ok=True, parents=True)
        Image.fromarray(viz3.astype(np.uint8)).save(f"{hparams.output_path}/viz/{file_name}.jpg")



            

    print('done')


if __name__ == '__main__':
    hello(_get_train_opts())

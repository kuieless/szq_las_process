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


torch.cuda.set_device(4)


def _get_train_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Campus_new',required=False, help='')
    parser.add_argument('--output_path', type=str, default='campus_gt_label/instances_gt_5_replace_instancelabel7',required=False, help='experiment name')
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

    project_instances = []
    for ext in ('*.png', '*.jpg'):
        project_instances.extend(glob(os.path.join('campus_gt_label/instances_gt_3_3dproject', ext)))
    project_instances.sort()
    project_instances = project_instances[7:8]


    i4_instances = []
    for ext in ('*.png', '*.jpg'):
        i4_instances.extend(glob(os.path.join('campus_gt_label/instances_gt_4_merge123', ext)))
    i4_instances.sort()
    i4_instances = i4_instances[7:8]

    H, W = 912,1368
    for idx, project_instance in enumerate(tqdm(project_instances)):
        file_name = Path(project_instance).stem
        i4_instance = i4_instances[idx]

        ###把投影图中 labels=7分成两栋的建筑)  覆盖   改过的 campus_gt_label/instances_gt_4_merge123

        i4 = np.array(Image.open(i4_instance))
        project3 = np.array(Image.open(project_instance))
        rgb = Image.open(os.path.join(rgb_path,f'{file_name}.jpg')).convert('RGB')
        size = rgb.size

        if size[0] != W or size[1] != H:
            rgb = rgb.resize((W, H), Image.LANCZOS)

        mask_ = i4 == 39
        mask_[:int(H*0.15),:] = False


        rgb = np.asarray(rgb)

        i4_new = i4.copy()
        new_building_mask = project3 == 7
        i4_new[new_building_mask] =7
        i4_new[mask_]=7
        
        


        Path(f"{hparams.output_path}").mkdir(exist_ok=True, parents=True)
        Image.fromarray(i4_new.astype(np.uint16)).save(f"{hparams.output_path}/{file_name}.png")

        alpha=0.65
        i4_viz = label_to_color(i4)*alpha+rgb*(1-alpha)
        i4_new_viz = label_to_color(i4_new)*alpha+rgb*(1-alpha)
        mask_viz = label_to_color(mask_)*255
        viz = np.concatenate([i4_viz, i4_new_viz, mask_viz],1)

        Path(f"{hparams.output_path}/viz").mkdir(exist_ok=True, parents=True)
        Image.fromarray(viz.astype(np.uint8)).save(f"{hparams.output_path}/viz/{file_name}.jpg")
        print(file_name)



            

    print('done')


if __name__ == '__main__':
    hello(_get_train_opts())

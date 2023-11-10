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



torch.cuda.set_device(4)


def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Campus_new',required=False, help='')
    parser.add_argument('--exp_name', type=str, default='logs_357/test',required=False, help='experiment name')
    parser.add_argument('--output_path', type=str, default='zyq/1109_change_label',required=False, help='experiment name')

    
    return parser.parse_args()


def hello(hparams: Namespace) -> None:
    hparams.ray_altitude_range = [-95, 54]
    hparams.dataset_type='memory_depth_dji_instance'
    hparams.depth_dji_type=='mesh'
    hparams.label_name = 'm2f' # ['m2f', 'merge', 'gt']
    hparams.instance_name = 'instances_mask_0.001' # ['m2f', 'merge', 'gt']
    hparams.sampling_mesh_guidance=True

    runner = Runner(hparams)
    train_items = runner.val_items



    for idx, metadata_item in enumerate(tqdm(train_items, desc="")):
        file_name = Path(metadata_item.image_path).stem
        
        device='cuda'
        overlap_threshold=0.5
        ###NOTE 需要将shuffle调成False, 不打乱，按照顺序处理
        H, W = metadata_item.H, metadata_item.W
       
        semantic_label = metadata_item.load_gt()
        instance_label = metadata_item.load_instance_gt()

        building_mask = semantic_label==1
        instance_label[~building_mask] = 0 
        instance_label[instance_label==31] =0
        instance_label[instance_label==42] =0
        instance_label[instance_label==26] =0



        Path(f"{hparams.output_path}").mkdir(exist_ok=True, parents=True)
        Image.fromarray(instance_label.cpu().numpy().astype(np.uint8)).save(f"{hparams.output_path}/{Path(metadata_item.label_path).stem}.png")

            

    print('done')


if __name__ == '__main__':
    hello(_get_train_opts())

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
    parser.add_argument('--pose_path', type=str, default='/data/yuqi/Datasets/DJI/Longhua_block1_20231009/val/metadata',required=False, help='')
    parser.add_argument('--save_path', type=str, default='output/longhua_block1_resize_metadata',required=False, help='experiment name')

    
    return parser.parse_args()


def hello(hparams: Namespace) -> None:
    pose_path = hparams.pose_path
    save_path = hparams.save_path

    Path(save_path).mkdir(exist_ok=True)

    used_files = []
    for ext in ('*.png', '*.pt'):
        used_files.extend(glob(os.path.join(pose_path, ext)))
    used_files.sort()
    # used_files=used_files[383:384]
    # H, W = 910, 1365
    H, W = 1024, 1536
    H_ori, W_ori = 5460, 8192
    H_scale = H_ori / H
    W_scale = W_ori / W


    
    for p_path in tqdm(used_files):
        
        metadata = torch.load(p_path, map_location='cpu')
        metadata['H'] = H
        metadata['W'] = W
        metadata['intrinsics'][0] = metadata['intrinsics'][0] / W_scale
        metadata['intrinsics'][1] = metadata['intrinsics'][1] / H_scale
        metadata['intrinsics'][2] = metadata['intrinsics'][2] / W_scale
        metadata['intrinsics'][3] = metadata['intrinsics'][3] / H_scale


        torch.save(metadata, os.path.join(save_path, f'{Path(p_path).name}'))

        
    
    print("done")




if __name__ == '__main__':
    hello(_get_train_opts())

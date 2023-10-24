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

from dji.visual_poses import load_poses
from pathlib import Path
from mega_nerf.ray_utils import get_rays, get_ray_directions
from PIL import Image



def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--instance_label_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/instances_gt',required=False, help='')
    parser.add_argument('--output_path', type=str, default='zyq/1024_remapping_instance',required=False, help='experiment name')
    
    return parser.parse_args()


def hello(hparams: Namespace) -> None:
    
    instance_label_path = hparams.instance_label_path
    output_path = hparams.output_path
    if not os.path.exists(output_path):
        Path(output_path).mkdir(parents=True)

    used_files = []
    for ext in ('*.png', '*.jpg'):
        used_files.extend(glob(os.path.join(instance_label_path, ext)))
    used_files.sort()

    for used_file in tqdm(used_files):
        file_name = Path(used_file).stem
        instance_labels = np.array(Image.open(used_file))

        # 11  12   16  18   25  26
        instance_labels[instance_labels==12]=0
        instance_labels[instance_labels==13]=0
        instance_labels[instance_labels==17]=0
        instance_labels[instance_labels==19]=0
        instance_labels[instance_labels==26]=0
        instance_labels[instance_labels==27]=0

        
        Image.fromarray(instance_labels.astype(np.uint16)).save(os.path.join(output_path, file_name +'.png'))
        

    

if __name__ == '__main__':
    hello(_get_train_opts())

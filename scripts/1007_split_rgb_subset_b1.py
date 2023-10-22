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

def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds',required=False, help='')
    parser.add_argument('--exp_name', type=str, default='logs_357/test',required=False, help='experiment name')
    
    return parser.parse_args()


def hello(hparams: Namespace) -> None:
    hparams.ray_altitude_range = [-186, -36]
    hparams.dataset_type='memory_depth_dji'



    runner = Runner(hparams)
    train_items = runner.train_items


    split_list=[]
    for metadata_item in tqdm(train_items):
        gt_label = metadata_item.load_gt()
        has_nonzero = (gt_label != 0).any()
        non_zero_ratio = torch.sum(gt_label != 0).item() / gt_label.numel()
        if has_nonzero and non_zero_ratio>0.1:
            split_list.append(f"{metadata_item.image_path.stem}.pt")
    
    

    file_name = "longhuab1_subset.txt"
    # 打开文件以写入模式（"w" 表示写入）
    with open(file_name, "w") as file:
        # 使用循环将列表中的元素写入文件
        for item in split_list:
            file.write(f'"{item}" ')
    
    print(len(split_list))




if __name__ == '__main__':
    hello(_get_train_opts())

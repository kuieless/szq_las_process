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



def calculate_entropy(labels):
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



def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926',required=False, help='')
    parser.add_argument('--exp_name', type=str, default='logs_357/test',required=False, help='experiment name')
    parser.add_argument('--project_path', type=str, default='logs_dji/1003_yingrenshi_density_depth_hash22_semantic/14/eval_200000_far0.3/1015_3d_get2dlabel_test_far0.3/project_far_to_ori',required=False, help='')
    parser.add_argument('--output_path', type=str, default='logs_dji/1003_yingrenshi_density_depth_hash22_semantic/14/eval_200000_far0.3/1015_3d_get2dlabel_test_far0.3/compare_vis',required=False, help='experiment name')
    # parser.add_argument('--output_path', type=str, default='zyq/1010_3d_get2dlabel_gt_test',required=False, help='experiment name')

    
    return parser.parse_args()


def hello(hparams: Namespace) -> None:


    project_path = hparams.project_path
    output_path = hparams.output_path
    if not os.path.exists(output_path):
        Path(output_path).mkdir(parents=True)
        
    if 'Longhua' in hparams.dataset_path:
        hparams.train_scale_factor =1
        hparams.val_scale_factor =1


    hparams.ray_altitude_range = [-95, 54]
    hparams.dataset_type='memory_depth_dji'
    device = 'cpu'
    hparams.label_name = 'm2f' # ['m2f', 'merge', 'gt']
    runner = Runner(hparams)
    train_items = runner.train_items



    # used_files = []
    # for ext in ('*.png', '*.jpg'):
    #     used_files.extend(glob(os.path.join(project_path, ext)))
    # used_files.sort()
    # process_item = [Path(far_p).stem for far_p in used_files]

    process_item=[]
    for metadata_item in tqdm(train_items):
        gt_label = metadata_item.load_gt()
        has_nonzero = (gt_label != 0).any()
        non_zero_ratio = torch.sum(gt_label != 0).item() / gt_label.numel()
        if has_nonzero and non_zero_ratio>0.1:
            process_item.append(f"{metadata_item.image_path.stem}")
    print(len(process_item))
    
    for metadata_item in tqdm(train_items, desc="extract the far m2f label"):
        file_name = Path(metadata_item.image_path).stem
        if file_name not in process_item or metadata_item.is_val:
            continue
        
        # 读取ori的标签文件
        ori_m2f = metadata_item.load_label()
        H, W = ori_m2f.shape


        # 读取far project to ori 的标签文件
        project_m2f = Image.open(os.path.join(project_path, file_name+'.png'))    #.convert('RGB')
        project_m2f = torch.ByteTensor(np.asarray(project_m2f))
        ori_m2f = remapping(ori_m2f)
        project_m2f = remapping(project_m2f)
        
        unequal_mask = torch.ne(ori_m2f, project_m2f)

        #### 获取不相等位置的索引
        # unequal_indices = torch.nonzero(unequal_mask, as_tuple=False)

        diff_ori = ori_m2f * unequal_mask
        diff_project = project_m2f * unequal_mask


        img = metadata_item.load_image()
        color_label1 = custom2rgb(ori_m2f.numpy())
        color_label1 = color_label1.reshape(H, W,3)
        merge1 = 0.7 * img.numpy() + 0.3 * color_label1

        color_label2 = custom2rgb(project_m2f.numpy())
        color_label2 = color_label2.reshape(H, W,3)
        merge2 = 0.7 * img.numpy() + 0.3 * color_label2


        color_label3 = custom2rgb(diff_ori.numpy())
        color_label3 = color_label3.reshape(H, W,3)
        merge3 = 0.5 * img.numpy() + 0.5 * color_label3

        color_label4 = custom2rgb(diff_project.numpy())
        color_label4 = color_label4.reshape(H, W,3)
        merge4 = 0.5 * img.numpy() + 0.5 * color_label4

        top_row = np.hstack((merge1, merge2))
        bottom_row = np.hstack((merge3, merge4))

        # 最终的2x2拼接图像
        final_image = np.vstack((top_row, bottom_row))

        Image.fromarray(final_image.astype(np.uint8)).save(os.path.join(output_path,f"{file_name}.jpg"))


    print('done')


if __name__ == '__main__':
    #meshMask = torch.zeros(912, 1368, 1)
    #projected_points = torch.rand(2222058, 2) * 500

    #x = projected_points[:, 0].long()
    #y = projected_points[:, 1].long()
    #mesh_depths = meshMask[x, y]
    #print(mesh_depths.shape)
    #import pdb; pdb.set_trace()

    hello(_get_train_opts())

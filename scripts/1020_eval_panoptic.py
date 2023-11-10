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
from tools.contrastive_lift.utils import cluster, visualize_panoptic_outputs
from torchvision.utils import make_grid
from gp_nerf.eval_utils import calculate_panoptic_quality_folders


torch.cuda.set_device(6)





def hello() -> None:
    # H,W = 912,1368
    # dataset_path='/data/yuqi/Datasets/DJI/Campus_new'
    # dataset_path='/data/yuqi/Datasets/DJI/Yingrenshi_20230926'

    H,W = 1024,1536
    dataset_path='/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds'
    # dataset_path='/data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds'




    path_target_sem = os.path.join(dataset_path, 'val', 'labels_gt')
    # path_target_inst = os.path.join(dataset_path, 'val', 'instances_gt_noremapping')
    path_target_inst = os.path.join(dataset_path, 'val', 'instances_gt')

    experiment_path_current = 'logs_longhua_b1/1108_longhua_b1_density_depth_hash22_instance_origin_sam_0.001_depth_crossview_all/0/eval_100000'
    path_pred_sem = os.path.join(experiment_path_current, 'pred_semantics')
    path_pred_inst = os.path.join(experiment_path_current, 'pred_surrogateid')

    if Path(path_target_inst).exists():
        pq, sq, rq, metrics_each, pred_areas, target_areas, zyq_TP, zyq_FP, zyq_FN = calculate_panoptic_quality_folders(path_pred_sem, path_pred_inst, 
                        path_target_sem, path_target_inst, image_size=[W,H])
        with (Path('instance.txt')).open('w') as f:
            f.write(f'\n\npred_areas\n')  

            for key, value in pred_areas.items():
                f.write(f"    {key}: {value}\n")
            f.write(f'\n\ntarget_areas\n')  
            for key, value in target_areas.items():
                f.write(f"    {key}: {value}\n")
            
            f.write(f'\n\nTP\n')  
            for item in zyq_TP:
                f.write(    f"{item}\n")
            
            f.write(f'\n\nFP\n')  
            for item in zyq_FP:
                f.write(    f"{item}\n")
            
            f.write(f'\n\nFN\n')  
            for item in zyq_FN:
                f.write(f"    {item}\n")


        # val_metrics['pq'] = pq
        # val_metrics['sq'] = sq
        # val_metrics['rq'] = rq
        for key in metrics_each['all']:
            avg_val = metrics_each['all'][key]
            message = ' {}: {}'.format(key, avg_val)
            print(message)

    print(f"pq, sq, rq: {pq}, {sq}, {rq}")
    print('done')


if __name__ == '__main__':

    hello()

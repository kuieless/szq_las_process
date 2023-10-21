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


torch.cuda.set_device(5)





def hello() -> None:
    device='cuda'
    Path(os.path.join('zyq','1020_panoptic')).mkdir(exist_ok=True)
    output_dir = 'logs_dji/1020_yingrenshi_density_depth_hash22_instance_freeze_gt/0/eval_90000/val_rgbs/panoptic'
    all_thing_features = np.load(os.path.join(output_dir, "all_thing_features.npy"))
    all_points_semantics = np.load(os.path.join(output_dir, "all_points_semantics.npy"))
    all_points_rgb = np.load(os.path.join(output_dir, "all_points_rgb.npy"))
    all_points_semantics=torch.from_numpy(all_points_semantics).to(device).view(15, 912*1368)
    all_points_rgb=torch.from_numpy(all_points_rgb).to(device).view(15, 912*1368, -1)

    thing_classes = [1]
    
    all_points_instances = cluster(all_thing_features, bandwidth=0.3, device=device, num_images=15)
    save_i=0
    # for p_rgb, p_semantics, p_instances in zip(all_points_rgb, all_points_semantics, all_points_instances)
    for save_i in range(15):
        p_rgb = all_points_rgb[save_i]
        p_semantics = all_points_semantics[save_i]
        p_instances = all_points_instances[save_i]

        stack = visualize_panoptic_outputs(
            p_rgb, p_semantics, p_instances, None, None, None, None,
            912, 1368, thing_classes=thing_classes, visualize_entropy=False
        )
        grid = make_grid(stack, value_range=(0, 1), normalize=True, nrow=5).permute((1, 2, 0)).contiguous()
        grid = (grid * 255).cpu().numpy().astype(np.uint8)
        
        Image.fromarray(grid).save(os.path.join('zyq','1020_panoptic',("%06d.jpg" % save_i)))

    
    print('done')


if __name__ == '__main__':

    hello()

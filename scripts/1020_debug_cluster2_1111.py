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


# torch.cuda.set_device(7)





def hello() -> None:
    H, W = 912, 1368
    eval_num = 11
    train_num = 0

    device='cuda'
    bandwidth=0.2
    num_points=200000
    use_dbscan=True
    use_silverman=False
    cluster_size=100
    output=f'1111_debug_campus_cluster'
    Path(os.path.join('zyq',output)).mkdir(exist_ok=True)
    Path(os.path.join('zyq',output,'pred_semantics')).mkdir(exist_ok=True)
    Path(os.path.join('zyq',output,'pred_surrogateid')).mkdir(exist_ok=True)
    Path(os.path.join('zyq',output,'gt_semantics')).mkdir(exist_ok=True)
    Path(os.path.join('zyq',output,'gt_surrogateid')).mkdir(exist_ok=True)


    # output_dir = 'logs_campus/1110_campus_density_depth_hash22_instance_origin_sam_0.001_depth_crossview_sam64/1/eval_100000/panoptic'
    output_dir = '/data/yuqi/code/GP-NeRF-semantic/logs_campus/1110_campus_density_depth_hash22_instance_origin_sam64/0/eval_100000/panoptic'
    all_thing_features = np.load(os.path.join(output_dir, "all_thing_features.npy"))
    all_points_semantics = np.load(os.path.join(output_dir, "all_points_semantics.npy"))
    all_points_rgb = np.load(os.path.join(output_dir, "all_points_rgb.npy"))
    all_points_semantics=torch.from_numpy(all_points_semantics).to(device).view(eval_num, H*W)
    all_points_rgb=torch.from_numpy(all_points_rgb).to(device).view(eval_num, H*W, -1)
    gt_points_rgb = np.load(os.path.join(output_dir, "gt_points_rgb.npy"))
    gt_points_semantic = np.load(os.path.join(output_dir, "gt_points_semantic.npy"))
    gt_points_instance = np.load(os.path.join(output_dir, "gt_points_instance.npy"))
    gt_points_semantic=torch.from_numpy(gt_points_semantic).to(device).view(eval_num, H*W)
    gt_points_rgb=torch.from_numpy(gt_points_rgb).to(device).view(eval_num, H*W, -1)
    gt_points_instance=torch.from_numpy(gt_points_instance).to(device).view(eval_num, H*W)

    thing_classes = [0,1,2,3,4]
    
    all_centroids=None
    
    # with open('logs_dji/1025_yingrenshi_density_depth_hash22_instance_freeze_gt_slow/0/eval_170000/test_centroids.npy', 'rb') as f:
    #     all_centroids = pickle.load(f)
    
    
    all_points_instances, centroids = cluster(all_thing_features, bandwidth=bandwidth, device=device, num_images=eval_num+train_num, 
                                   num_points=num_points, use_silverman=use_silverman, use_dbscan=use_dbscan,cluster_size=cluster_size, all_centroids=all_centroids)
    print(f"centroids shape: {centroids.shape}")
    save_i=0
    # for p_rgb, p_semantics, p_instances in zip(all_points_rgb, all_points_semantics, all_points_instances)
    for save_i in range(eval_num):
        p_rgb = all_points_rgb[save_i]
        # p_semantics = all_points_semantics[save_i]
        p_semantics = gt_points_semantic[save_i]

        p_instances = all_points_instances[save_i]

        gt_rgb = gt_points_rgb[save_i]
        gt_semantics = gt_points_semantic[save_i]
        gt_instances = gt_points_instance[save_i]

        output_semantics_with_invalid = p_semantics.detach()
        Image.fromarray(output_semantics_with_invalid.reshape(H, W).cpu().numpy().astype(np.uint8)).save(
            os.path.join('zyq', output , 'pred_semantics', ("%06d.png" % save_i)))
        
        Image.fromarray(p_instances.argmax(dim=1).reshape(H, W).cpu().numpy().astype(np.uint16)).save(
            os.path.join('zyq', output , 'pred_surrogateid', ("%06d.png" % save_i)))

        

        Image.fromarray(gt_semantics.reshape(H, W).cpu().numpy().astype(np.uint8)).save(
            os.path.join('zyq', output , 'gt_semantics', ("%06d.png" % save_i)))

        
        Image.fromarray(gt_instances.reshape(H, W).cpu().numpy().astype(np.uint16)).save(
            os.path.join('zyq', output , 'gt_surrogateid', ("%06d.png" % save_i)))        
        
        stack = visualize_panoptic_outputs(
            p_rgb, p_semantics, p_instances, None, gt_rgb, gt_semantics, gt_instances,
            H, W, thing_classes=thing_classes, visualize_entropy=False
        )
        grid = make_grid(stack, value_range=(0, 1), normalize=True, nrow=3).permute((1, 2, 0)).contiguous()
        grid = (grid * 255).cpu().numpy().astype(np.uint8)
        
        Image.fromarray(grid).save(os.path.join('zyq',output,("%06d.jpg" % save_i)))

    
    path_target_sem = os.path.join('zyq',output,'gt_semantics')
    path_target_inst = os.path.join('zyq',output,'gt_surrogateid')
    path_pred_sem = os.path.join('zyq',output,'pred_semantics')
    path_pred_inst = os.path.join('zyq',output,'pred_surrogateid')
    if Path(path_target_inst).exists():
        pq, sq, rq, metrics_each, pred_areas, target_areas, zyq_TP, zyq_FP, zyq_FN  = calculate_panoptic_quality_folders(path_pred_sem, path_pred_inst, 
                        path_target_sem, path_target_inst, image_size=[W,H])
        # val_metrics['pq'] = pq
        # val_metrics['sq'] = sq
        # val_metrics['rq'] = rq
        for key in metrics_each['all']:
            avg_val = metrics_each['all'][key]
            message = ' {}: {}'.format(key, avg_val)
            print(message)

        print(f'\n pq, sq, rq: {pq:.5f} {sq:.5f} {rq:.5f}\n')  
    
        with(Path(os.path.join('zyq',output, 'metric.txt'))).open('w') as f:
            f.write(f'\n pq, sq, rq: {pq:.5f} {sq:.5f} {rq:.5f}\n')  
        print('done')


if __name__ == '__main__':

    hello()

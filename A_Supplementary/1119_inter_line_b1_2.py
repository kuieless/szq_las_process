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


def inter_two_poses(pose_a, pose_b, alpha):
    ret = np.zeros([3, 4], dtype=np.float64)
    rot_a = R.from_matrix(pose_a[:3, :3])
    rot_b = R.from_matrix(pose_b[:3, :3])
    key_rots = R.from_matrix(np.stack([pose_a[:3, :3], pose_b[:3, :3]], 0))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    rot = slerp(1. - alpha)
    ret[:3, :3] = rot.as_matrix()
    ret[:3, 3] = (pose_a * alpha + pose_b * (1. - alpha))[:3, 3]
    return ret


def inter_poses(key_poses, n_out_poses, sigma=1.):
    n_key_poses = len(key_poses)
    out_poses = []

    # Ensure that there are at least two key poses
    if n_key_poses < 2:
        raise ValueError("At least two key poses are required for interpolation.")
    # Interpolate between the first and last poses
    slerp = Slerp([0, 1], R.from_matrix(np.stack([key_poses[0][:3, :3], key_poses[-1][:3, :3]], axis=0)))   
    for i in range(n_out_poses):
        alpha = i / (n_out_poses - 1)  # Vary alpha from 0 to 1
        interpolated_rotation = slerp(alpha).as_matrix()
        cur_pose = np.zeros_like(key_poses[0])
        cur_pose[:3, :3] = interpolated_rotation
        cur_pose[:3, 3] = key_poses[0][:3, 3] * (1 - alpha) + key_poses[-1][:3, 3] * alpha
        out_poses.append(cur_pose)
    return np.stack(out_poses)



from dji.visual_poses import load_poses
from pathlib import Path
from mega_nerf.ray_utils import get_rays, get_ray_directions

@click.command()
@click.option('--data_dir', type=str, default='/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds')
@click.option('--key_poses', type=str, default='129,408')
@click.option('--n_out_poses', type=int, default=60)

def hello(data_dir, n_out_poses, key_poses):

    # poses = np.load(pjoin(data_dir, 'cams_meta.npy')).reshape(-1, 27)[:, :12].reshape(-1, 3, 4)    #(N,3,4)
    poses = np.array(load_poses(data_dir))  
    
    train_file_names = sorted(list((Path(data_dir) / 'train' / 'metadata').iterdir()))
    val_file_names = sorted(list((Path(data_dir) / 'val' / 'metadata').iterdir()))
    all_file_names = train_file_names + val_file_names
    # 根据文件名中的数字进行排序
    metadata_paths = sorted(all_file_names, key=lambda x: int(x.stem))

    

    n_poses = len(poses)
   
    key_poses = np.array([int(_) for _ in key_poses.split(',')])
    key_poses_1 = poses[key_poses]

    out_poses = inter_poses(key_poses_1, n_out_poses)


    out_poses = np.ascontiguousarray(out_poses.astype(np.float64))

    source_file = metadata_paths[0]

    metadata_old = torch.load(source_file)

    if not os.path.exists(os.path.join('Output_subset', 'render', 'metadata')):
        Path(os.path.join('Output_subset', 'render', 'metadata')).mkdir(parents=True, exist_ok=True)

    for i in range(len(out_poses)):
        metadata_old['c2w'] = torch.FloatTensor(out_poses[i])
        torch.save(metadata_old, os.path.join('Output_subset', 'render', 'metadata', f"{i:06d}.pt"))


    np.save(pjoin(data_dir, 'poses_render.npy'), out_poses)
    print('Done')


if __name__ == '__main__':
    hello()

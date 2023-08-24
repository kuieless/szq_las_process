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
    for i in range(n_out_poses):
        w = np.linspace(0, n_key_poses - 1, n_key_poses)
        w = np.exp(-(np.abs(i / n_out_poses * n_key_poses - w) / sigma)**2)
        w = w + 1e-6
        w /= np.sum(w)
        cur_pose = key_poses[0]
        cur_w = w[0]
        for j in range(0, n_key_poses - 1):
            cur_pose = inter_two_poses(cur_pose, key_poses[j + 1], cur_w / (cur_w + w[j + 1]))
            cur_w += w[j + 1]

        out_poses.append(cur_pose)

    return np.stack(out_poses)

from dji.visual_poses import load_poses
from pathlib import Path
from mega_nerf.ray_utils import get_rays, get_ray_directions


@click.command()
@click.option('--data_dir', type=str, default='/data/yuqi/Datasets/MegaNeRF/residence_subset')
@click.option('--key_poses', type=str, default='96')
@click.option('--n_out_poses', type=int, default=20)

def hello(data_dir, n_out_poses, key_poses):

    # poses = np.load(pjoin(data_dir, 'cams_meta.npy')).reshape(-1, 27)[:, :12].reshape(-1, 3, 4)    #(N,3,4)
    poses = np.array(load_poses(data_dir))  
    
    metadata_paths = sorted(list((Path(data_dir) / 'train' / 'metadata').iterdir()))

    n_poses = len(poses)
   
    key_poses = np.array([int(_) for _ in key_poses.split(',')])
    key_poses_1 = poses[key_poses]


    coordinate_info = torch.load(Path(data_dir) / 'coordinates.pt', map_location='cpu')
    pose_scale_factor = coordinate_info['pose_scale_factor']

    metadata_item = torch.load(metadata_paths[key_poses[0]])
    select_pose = metadata_item['c2w']
    
    directions = get_ray_directions(metadata_item['W'],
                                    metadata_item['H'],
                                    metadata_item['intrinsics'][0],
                                    metadata_item['intrinsics'][1],
                                    metadata_item['intrinsics'][2],
                                    metadata_item['intrinsics'][3],
                                    True,
                                    torch.device('cpu'))
    image_rays = get_rays(directions, metadata_item['c2w'], 0, 1e5, [30, 118])
    ray_d = image_rays[int(metadata_item['H']/2), int(metadata_item['W']/2), 3:6]
    ray_o = image_rays[int(metadata_item['H']/2), int(metadata_item['W']/2), :3]
    near = image_rays[int(metadata_item['H']/2), int(metadata_item['W']/2), 6] /pose_scale_factor
    far = image_rays[int(metadata_item['H']/2), int(metadata_item['W']/2), 7] /pose_scale_factor

    
    z_vals_inbound = near * 0 + far * 1
    
    new_o = ray_o + ray_d * z_vals_inbound

    new_pose = poses[key_poses]
    new_pose[:,:,3]= new_o.numpy()
    key_poses_1 = np.concatenate([poses[key_poses], new_pose], 0)

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


if __name__ == '__main__':
    hello()

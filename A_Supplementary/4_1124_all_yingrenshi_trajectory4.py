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
import math


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


def _get_train_opts():
    parser = get_opts_base()
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926',required=False, help='')
    parser.add_argument('--exp_name', type=str, default='logs_357/test',required=False, help='experiment name')
    
    return parser.parse_args()



@click.command()
@click.option('--data_dir', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926')
@click.option('--key_poses', type=str, default='96,121,301,303')
@click.option('--n_out_poses', type=int, default=120)

def hello(data_dir, n_out_poses, key_poses):

    hparams=_get_train_opts()
    ###读入一些相机参数

    hparams.ray_altitude_range = [-95, 54]
    hparams.dataset_type='memory_depth_dji'
    
    runner = Runner(hparams)
    train_items = runner.train_items


    # poses = np.load(pjoin(data_dir, 'cams_meta.npy')).reshape(-1, 27)[:, :12].reshape(-1, 3, 4)    #(N,3,4)
    poses = np.array(load_poses(data_dir))  
    
    train_file_names = sorted(list((Path(data_dir) / 'train' / 'metadata').iterdir()))
    val_file_names = sorted(list((Path(data_dir) / 'val' / 'metadata').iterdir()))
    all_file_names = train_file_names + val_file_names
    # 根据文件名中的数字进行排序
    metadata_paths = sorted(all_file_names, key=lambda x: int(x.stem))

    

    n_poses = len(poses)
   
    key_poses = np.array([int(_) for _ in key_poses.split(',')])
    key_poses_all = poses[key_poses]


    
    ###  0. 设定第一帧的位置
    key1= key_poses_all[0]

    ###  1. 拉远far视角
    ## 接下来找far视角
    metadata_item=train_items[key_poses[0]]
    directions = get_ray_directions(metadata_item.W,
                                    metadata_item.H,
                                    metadata_item.intrinsics[0],
                                    metadata_item.intrinsics[1],
                                    metadata_item.intrinsics[2],
                                    metadata_item.intrinsics[3],
                                    True,
                                    torch.device('cpu'))
    image_rays = get_rays(directions, metadata_item.c2w, 0, 1e5, [30, 118])
    ray_d = image_rays[int(metadata_item.H/2), int(metadata_item.W/2), 3:6]
    ray_o = image_rays[int(metadata_item.H/2), int(metadata_item.W/2), :3]

    # z_vals_inbound = gt_depths_valid.min() * 1.5
    z_vals_inbound = 0.3
    new_o = ray_o - ray_d * z_vals_inbound
    key2 = metadata_item.c2w
    ## 先拔高
    key2[:,3]= new_o
    traj1 = np.concatenate([key1[np.newaxis,:],key2[np.newaxis,:]],0)
    out_poses1 = inter_poses(traj1, 10)



    #### 2. 旋转视角
    def rad(x):
        return math.radians(x)
    angle=-25
    cosine = math.cos(rad(angle))
    sine = math.sin(rad(angle))
    rotation_matrix_x = torch.tensor([[1, 0, 0],
                    [0, cosine, sine],
                    [0, -sine, cosine]])
    angle=0
    cosine = math.cos(rad(angle))
    sine = math.sin(rad(angle))
    rotation_matrix_y = torch.tensor([[cosine, 0, sine],
                    [0, 1, 0],
                    [-sine, 0, cosine]])

    key3=key2.clone()
    key3[:3,:3]=rotation_matrix_y @ (rotation_matrix_x @ key3[:3,:3])

    traj2 = np.concatenate([key2[np.newaxis,:],key3[np.newaxis,:]],0)
    out_poses2 = inter_poses(traj2, 7)


    ##  这里定义一个正向的角度，后面使用
    angle=-35
    cosine = math.cos(rad(angle))
    sine = math.sin(rad(angle))
    rotation_matrix_x = torch.tensor([[1, 0, 0],
                    [0, cosine, sine],
                    [0, -sine, cosine]])
    rotation_1 = rotation_matrix_y @ (rotation_matrix_x @ key2[:3,:3])


    ####3 . 俯冲
    key4=key_poses_all[1]
    key4[0,3]=key4[0,3]+ 0.5*(key2[0,3]-key4[0,3])
    key4[:3,:3]=rotation_1
    traj3 = np.concatenate([key3[np.newaxis,:],key4[np.newaxis,:]],0)
    out_poses3 = inter_poses(traj3, 15)

    ### 4. 往前飞
    ## -60比较正  -- 6.mp4
    angle=-90
    cosine = math.cos(rad(angle))
    sine = math.sin(rad(angle))
    rotation_matrix_x = torch.tensor([[1, 0, 0],
                    [0, cosine, sine],
                    [0, -sine, cosine]])
    rotation_2 = rotation_matrix_y @ (rotation_matrix_x @ key2[:3,:3])

    key5=key_poses_all[2]
    key5[:3,:3]=rotation_2
    traj4 = np.concatenate([key4[np.newaxis,:],key5[np.newaxis,:]],0)
    out_poses4 = inter_poses(traj4, 15)


    
    # ### 5. 掉头
    # ## -60比较正  -- 6.mp4
    # angle=-240
    # cosine = math.cos(rad(angle))
    # sine = math.sin(rad(angle))
    # rotation_matrix_x = torch.tensor([[1, 0, 0],
    #                 [0, cosine, sine],
    #                 [0, -sine, cosine]])
    # rotation_2 = rotation_matrix_y @ (rotation_matrix_x @ key2[:3,:3])

    # key6=key_poses_all[3]
    # key6[:3,:3]=rotation_2
    # traj5 = np.concatenate([key5[np.newaxis,:],key6[np.newaxis,:]],0)
    # out_poses5 = inter_poses(traj5, 15)




    # out_poses = np.concatenate([out_poses1, out_poses2, out_poses3, out_poses4],0)
    # out_poses = np.concatenate([out_poses1,out_poses2,out_poses3, out_poses4],0)
    out_poses = np.concatenate([out_poses3, out_poses4],0)


    out_poses = np.ascontiguousarray(out_poses.astype(np.float64))


    source_file = metadata_paths[0]

    metadata_old = torch.load(source_file)


    if not os.path.exists(os.path.join(data_dir, 'render_supp_tra', 'metadata')):
        Path(os.path.join(data_dir, 'render_supp_tra', 'metadata')).mkdir(parents=True, exist_ok=True)

    for i in range(len(out_poses)):
        metadata_old['c2w'] = torch.FloatTensor(out_poses[i])
        torch.save(metadata_old, os.path.join(data_dir, 'render_supp_tra', 'metadata', f"{i:06d}.pt"))


    np.save(pjoin(data_dir, 'poses_render.npy'), out_poses)
    print('Done')


if __name__ == '__main__':
    hello()

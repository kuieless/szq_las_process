## 这个代码用来获取 dji 数据集中的subdataset
## 选择了某一建筑物上的单点，将其投影到其他视角中，获取一个子集

import glob
import os
import numpy as np
import re
import shutil
from pathlib import Path
from dji.visual_poses import load_poses
import torch
import shutil
from PIL import Image
from gp_nerf.runner_gpnerf import Runner
from gp_nerf.opts import get_opts_base
from argparse import Namespace
from mega_nerf.ray_utils import get_ray_directions
import cv2
from tqdm import tqdm


def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--exp_name', type=str, default='logs_357/test',required=False, help='experiment name')
    parser.add_argument('--dataset_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan',type=str, required=False)
    
    return parser.parse_args()

def main(hparams: Namespace) -> None:

    occluded_threshold = 0.01
    hparams.ray_altitude_range = [-138, -35]
    hparams.depth_dji_type ='mesh'
    hparams.dataset_type='memory_depth_dji'
    hparams.dataset_type='memory_depth_dji'


    runner = Runner(hparams)
    train_items = runner.train_items
    val_items = runner.val_items
    W = train_items[0].W
    H = train_items[0].H
    device = torch.device('cpu')
    directions = get_ray_directions(train_items[0].W,
                                    train_items[0].H,
                                    train_items[0].intrinsics[0],
                                    train_items[0].intrinsics[1],
                                    train_items[0].intrinsics[2],
                                    train_items[0].intrinsics[3],
                                    train_items[0],
                                    device)
    depth_scale = torch.abs(directions[:, :, 2])
    

    #选取某张图片上的某个点
    select_index = 959
    select_item = train_items[select_index]
    rgbs = select_item.load_image()

    # 读取depth
    depth_map = select_item.load_depth_dji().view(H, W)
    depth_map = (depth_map * depth_scale)
    x, y = int(W * 1/2), int(H * 1/2)
    depth = depth_map[y, x].numpy()
    K1 = select_item.intrinsics
    K1 = np.array([[K1[0], 0, K1[2]],[0, K1[1], K1[3]],[0,0,1]])
    E1 = np.array(select_item.c2w)
    E1 = np.stack([E1[:, 0], E1[:, 1]*-1, E1[:, 2]*-1, E1[:, 3]], 1)
    pt_3d = np.dot(np.linalg.inv(K1), np.array([x, y, 1])) * depth
    pt_3d = np.append(pt_3d, 1)
    world_point = np.dot(np.concatenate((E1, [[0,0,0,1]]), 0), pt_3d)
    
    visualization=True

    save_dir = f'zyq/project_subset_{select_index}'
    Path(save_dir).parent.mkdir(exist_ok=True)
    Path(save_dir).mkdir(exist_ok=True)
    (Path(save_dir) / "sample").mkdir(exist_ok=True)

    if visualization:
    
        img= cv2.imread(f"{str(select_item.image_path)}")
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H))
        pt2 = (int(x), int(y))
        radius = 5
        color = (0, 0, 255)
        thickness = 2
        cv2.circle(img, pt2, radius, color, thickness)
        cv2.imwrite(f"{save_dir}/{select_item.image_path.stem}.jpg", img)


    for i, metadata_item in tqdm(enumerate(train_items)):

        E2 = np.array(metadata_item.c2w)
        E2 = np.stack([E2[:, 0], E2[:, 1]*-1, E2[:, 2]*-1, E2[:, 3]], 1)
        w2c = np.linalg.inv(np.concatenate((E2, [[0,0,0,1]]), 0))
        pt_3d_trans = np.dot(w2c, world_point)
        pt_2d_trans = np.dot(K1, pt_3d_trans[:3]) 
        pt_2d_trans = pt_2d_trans / pt_2d_trans[2]


        x2, y2 = int(pt_2d_trans[0]), int(pt_2d_trans[1])
        if x2 >= 0 and x2 < W and y2 >= 0 and y2 < H:
            depth_map2 = metadata_item.load_depth_dji().view(H, W)
            depth_map2 = (depth_map2 * depth_scale).numpy()
            depth2 = depth_map2[y2, x2]
            depth_diff = np.abs(depth2 - pt_3d_trans[2])
            pt2 = (int(pt_2d_trans[0]), int(pt_2d_trans[1]))
            
            if visualization:
                img = cv2.imread(str(metadata_item.image_path))
                if img.shape[1] != W or img.shape[0] != H:
                    img = cv2.resize(img, (W, H))

                radius = 5
                color = (0, 0, 255)
                thickness = 2

                if depth_diff > occluded_threshold:
                    color = (0, 255, 0)
                    img = cv2.circle(img, pt2, radius, color, thickness)
                    print(f'occluded points!!   the depth_diff: {depth_diff}')
                    print(f"{save_dir}/{metadata_item.image_path.stem}_occluded_{depth_diff}.jpg")
                    # cv2.imwrite(f"{save_dir}/{metadata_item.image_path.stem}_occluded_{depth_diff}.jpg", img)
                    cv2.imwrite(f"{save_dir}/sample/{metadata_item.image_path.stem}_project.jpg", img)

                else:
                    img = cv2.circle(img, pt2, radius, color, thickness)
                    print(f"{save_dir}/sample/{metadata_item.image_path.stem}_project.jpg   the depth_diff: {depth_diff}")

                    cv2.imwrite(f"{save_dir}/sample/{metadata_item.image_path.stem}_project.jpg", img)



if __name__ == '__main__':
    main(_get_train_opts())
    
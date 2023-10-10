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
from tools.unetformer.uavid2rgb import rgb2custom, custom2rgb
from PIL import Image
from pathlib import Path
import open3d as o3d
import pickle

def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926',required=False, help='')
    parser.add_argument('--exp_name', type=str, default='logs_357/test',required=False, help='experiment name')
    parser.add_argument('--output_path', type=str, default='zyq/1010_3d_get2dlabel',required=False, help='experiment name')

    
    return parser.parse_args()


def hello(hparams: Namespace) -> None:
    hparams.ray_altitude_range = [-95, 54]
    hparams.dataset_type='memory_depth_dji'
    device = 'cuda'

    output_path = hparams.output_path
    if not os.path.exists(output_path):
        Path(output_path).mkdir(parents=True)
    if not os.path.exists(os.path.join(output_path, 'vis')):
        Path(os.path.join(output_path, 'vis')).mkdir(parents=True)


    point_cloud = o3d.io.read_point_cloud("zyq/2d-3d-2d_yingrenshi_gt/point_cloud_50.ply")

    points_nerf = np.asarray(point_cloud.points)
    points_colors = np.asarray(point_cloud.colors)


    points_nerf = torch.from_numpy(points_nerf).to(device)
    points_nerf = points_nerf.float()
    print(points_nerf.shape)

    hparams.label_name = 'm2f' # ['m2f', 'merge', 'gt']

    runner = Runner(hparams)
    train_items = runner.train_items
    print(f"len train_items: {len(train_items)}")

    split_list=[]
    point_label_dict = {}
    point_label_list = []
    for i in range(points_nerf.shape[0]):
        point_label_list.append([])
    for metadata_item in tqdm(train_items):

        if hparams.label_name == 'gt':
            gt_label = metadata_item.load_gt()
        else:
            gt_label = metadata_item.load_label().to(device)

        
        H, W = metadata_item.H, metadata_item.W
        camera_rotation = metadata_item.c2w[:3,:3].to(device)
        camera_position = metadata_item.c2w[:3, 3].to(device)
        
        camera_matrix = torch.tensor([[metadata_item.intrinsics[0], 0, metadata_item.intrinsics[2]],
                              [0, metadata_item.intrinsics[1], metadata_item.intrinsics[3]],
                              [0, 0, 1]]).to(device)

        # NOTE: 2. 自己写，正确
        E2 = torch.hstack((camera_rotation, camera_position.unsqueeze(-1)))
        E2 = torch.stack([E2[:, 0], E2[:, 1]*-1, E2[:, 2]*-1, E2[:, 3]], 1).to(device)
        w2c = torch.inverse(torch.cat((E2, torch.tensor([[0, 0, 0, 1]]).to(device)), dim=0))
        points_homogeneous = torch.cat((points_nerf, torch.ones((points_nerf.shape[0], 1), dtype=torch.float32).to(device)), dim=1)
        pt_3d_trans = torch.mm(w2c, points_homogeneous.t())
        
        pt_2d_trans = torch.mm(camera_matrix, pt_3d_trans[:3])
        pt_2d_trans = pt_2d_trans / pt_2d_trans[2]
        projected_points = pt_2d_trans[:2].t()

        large_int = 1e6
        image_width, image_height = int(W), int(H)


        # 将 NumPy 数组转换为 PyTorch 张量，并移动到 GPU 上
        image = torch.zeros((image_height, image_width)).to(device)
        image_color = torch.zeros((image_height, image_width), dtype=torch.uint8).to(device)

        depth_map = large_int * torch.ones((image_height, image_width, 1), dtype=torch.uint8).to(device)
        depth_map_expand = large_int * torch.ones((image_height, image_width, 1), dtype=torch.uint8).to(device)

        # 获得落在图像上的点
        mask_x = (projected_points[:, 0] >= 0) & (projected_points[:, 0] <= image_width)
        mask_y = (projected_points[:, 1] >= 0) & (projected_points[:, 1] <= image_height)
        mask = mask_x & mask_y

        depth_z = pt_3d_trans[2]
        
        projected_points_mask = projected_points[mask, :]
        depth_z_mask = depth_z[mask]

        step = 2
        expand=True
        for x, y, depth, idx in tqdm(zip(projected_points_mask[:, 0], projected_points_mask[:, 1], depth_z_mask, mask.nonzero().squeeze(-1))):
            x, y = int(x), int(y)
            # if depth < depth_map[y, x]:
            #     depth_map[y, x] = depth
            #     image[y, x] = color[::-1]
            if expand and depth < depth_map_expand[y, x]:
                depth_map_expand[max(0, y - step):min(image_height, y + step), max(0, x - step):min(image_width, x + step)] = depth
                image[y, x] = idx
                image_color[y, x] = gt_label[y, x]
        image_color = custom2rgb(image_color.cpu().numpy())
        
        Image.fromarray(image_color.astype(np.uint8)).save(os.path.join(output_path, 'vis', f"{metadata_item.image_path.stem}.png"))
        
        for h in range(H):
            for w in range(W):
                point_label_list[int(image[h, w])].append(int(gt_label[h, w]))


        # for h in range(H):
        #     for w in range(W):
        #         point_label_dict[int(image[h, w])].append(gt_label[h, w])


    


    print('write pickle')
    # 将列表保存为二进制文件
    with open(f'{output_path}/point_label_list.pkl', 'wb') as file:
        pickle.dump(point_label_list, file)
        

    print("done")




if __name__ == '__main__':
    hello(_get_train_opts())

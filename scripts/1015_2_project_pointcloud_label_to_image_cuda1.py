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

from multiprocessing import Pool



def process_sample(args):
    metadata_item, points_nerf, points_color, threshold, device, hparams = args
    metadata_item = metadata_item[0]

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




    # 创建空白图像
    large_int = 1e6
    image_width, image_height = int(W), int(H)


    image = torch.zeros((image_height, image_width, 3)).to(device)
    image_expand = torch.zeros((image_height, image_width, 3)).to(device)
    image_expand_depth = torch.zeros((image_height, image_width, 3)).to(device)


    depth_map = large_int * torch.ones((image_height, image_width, 1), dtype=torch.uint8).to(device)
    depth_map_expand = large_int * torch.ones((image_height, image_width, 1), dtype=torch.uint8).to(device)


        # 获得落在图像上的点
    mask_x = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < image_width)
    mask_y = (projected_points[:, 1] >= 0) & (projected_points[:, 1] < image_height)
    mask = mask_x & mask_y
    # mask[::10]=False


    step = 1
    expand=True
    meshMask = metadata_item.load_depth_dji()

    x = projected_points[:, 0].long()
    y = projected_points[:, 1].long()
    x[~mask] = 0
    y[~mask] = 0
    mesh_depths = meshMask[y, x]
    mesh_depths[~mask] = -10

    depth_z = pt_3d_trans[2]
    mask_z = depth_z < (mesh_depths[:, 0] + threshold)
    mask_xyz = mask & mask_z

    projected_points_mask = projected_points[mask_xyz, :]
    points_color_mask = points_color[mask_xyz]


    for x, y, depth, color in tqdm(zip(projected_points_mask[:, 0], projected_points_mask[:, 1], depth_z[mask_xyz], points_color_mask)):
        x, y = int(x), int(y)

        # if depth < depth_map[y, x]:
        thresh = meshMask[y, x] + threshold
        if depth < thresh:
            depth_map[y, x] = depth
            image[y, x] = torch.flip(color, [0])
            image_expand[max(0, y - step):min(image_height, y + step), max(0, x - step):min(image_width, x + step)] = torch.flip(color, [0])
        if expand and depth < depth_map_expand[y, x] and depth < thresh:
            depth_map_expand[max(0, y - step):min(image_height, y + step), max(0, x - step):min(image_width, x + step)] = depth
            image_expand_depth[y, x] = torch.flip(color, [0])


    image = image.cpu().numpy()[:, :, ::-1]
    image_label = rgb2custom(image)

    image_expand_depth = image_expand_depth.cpu().numpy()[:, :, ::-1]
    image_expand_depth_label = rgb2custom(image_expand_depth)

    image_expand = image_expand.cpu().numpy()[:, :, ::-1]
    image_expand_label = rgb2custom(image_expand)


    



    Image.fromarray(image_label.astype(np.uint16)).save(os.path.join(hparams.output_path, 'labels_pc', f"{metadata_item.image_path.stem}.png"))
    Image.fromarray(image.astype(np.uint8)).save(os.path.join(hparams.output_path, 'vis', f"{metadata_item.image_path.stem}.png"))
    gt_label_rgb = custom2rgb(image_label)
    Image.fromarray(gt_label_rgb.astype(np.uint8)).save(os.path.join(hparams.output_path, 'val_vis', f"{metadata_item.image_path.stem}.png"))


    Image.fromarray(image_expand_depth_label.astype(np.uint16)).save(os.path.join(hparams.output_path, 'expand_depth_labels_pc', f"{metadata_item.image_path.stem}.png"))
    Image.fromarray(image_expand_depth.astype(np.uint8)).save(os.path.join(hparams.output_path, 'expand_depth_vis', f"{metadata_item.image_path.stem}.png"))


    Image.fromarray(image_expand_label.astype(np.uint16)).save(os.path.join(hparams.output_path, 'expand_labels_pc', f"{metadata_item.image_path.stem}.png"))
    Image.fromarray(image_expand.astype(np.uint8)).save(os.path.join(hparams.output_path, 'expand_vis', f"{metadata_item.image_path.stem}.png"))




def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926',required=False, help='')
    parser.add_argument('--exp_name', type=str, default='logs_357/test',required=False, help='experiment name')
    parser.add_argument('--output_path', type=str, default='zyq/1015_300image_pc2label',required=False, help='experiment name')
    parser.add_argument('--load_ply_path', type=str, default='/data/yuqi/code/GP-NeRF-semantic/zyq/1015_3d_get2dlabel_far0.3_project_300_image/results.ply',required=False, help='experiment name')


    return parser.parse_args()


    
def hello(hparams: Namespace) -> None:
    torch.multiprocessing.set_start_method('spawn')

    if not os.path.exists(hparams.output_path):
        Path(hparams.output_path).mkdir(parents=True)
        Path(os.path.join(hparams.output_path, 'vis')).mkdir(parents=True)
        Path(os.path.join(hparams.output_path, 'val_vis')).mkdir(parents=True)
        Path(os.path.join(hparams.output_path, 'labels_pc')).mkdir(parents=True)
        Path(os.path.join(hparams.output_path, 'expand_vis')).mkdir(parents=True)
        Path(os.path.join(hparams.output_path, 'expand_labels_pc')).mkdir(parents=True)
        Path(os.path.join(hparams.output_path, 'expand_depth_vis')).mkdir(parents=True)
        Path(os.path.join(hparams.output_path, 'expand_depth_labels_pc')).mkdir(parents=True)
        

    hparams.ray_altitude_range = [-95, 54]
    hparams.dataset_type='memory_depth_dji'
    
    device = 'cpu' #'cuda'
    threshold=0.015
    print('read point cloud')

    point_cloud = o3d.io.read_point_cloud(hparams.load_ply_path)
    points_nerf = np.asarray(point_cloud.points)
    points_color = np.asarray(point_cloud.colors)*255

    # load_ply_path = hparams.load_ply_path
    # load_data = np.genfromtxt(load_ply_path, usecols=(0, 1, 2, 3, 4, 5))
    # print(f'load_data: {load_data.shape}')
    # points_nerf = load_data[:,0:3]
    # points_color = load_data[:,3:6]


    points_nerf = torch.from_numpy(points_nerf).to(device)
    points_nerf = points_nerf.float()
    print(points_nerf.shape)
    points_color = torch.from_numpy(points_color).to(device)


    runner = Runner(hparams)
    val_items = runner.val_items
    print(f"len train_items: {len(val_items)}")

    num_parts = len(val_items)
    args_list = []
    for i in range(num_parts):
        start_idx = i
        end_idx = i + 1 
        block_item = val_items[start_idx:end_idx]
        args_list.append((block_item, points_nerf, points_color, threshold, device, hparams))

    num_processes = len(val_items)

    with Pool(num_processes) as pool:
        print("Pool is started.")
        pool.map(process_sample, args_list)
        print('ok')



if __name__ == '__main__':
    hello(_get_train_opts())

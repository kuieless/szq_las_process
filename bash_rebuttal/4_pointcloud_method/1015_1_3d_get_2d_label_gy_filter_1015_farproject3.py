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

# torch.cuda.set_device(6)


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
    parser.add_argument('--exp_name', type=str, default='logs_357/test',required=False, help='experiment name')
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926',required=False, help='')
    parser.add_argument('--output_path', type=str, default='zyq/1015_3d_get2dlabel_far0.3_project_3',required=False, help='experiment name')
    parser.add_argument('--far_project_m2f_paths', type=str, default='/data/yuqi/code/GP-NeRF-semantic/zyq/1015_far0.3_all/project_far_to_ori',required=False, help='')
    parser.add_argument('--metaXml_path', default='/data/yuqi/Datasets/DJI/origin/Yingrenshi_20230926_origin/terra_point_ply/metadata.xml', type=str)
    parser.add_argument('--load_ply_path', default='/data/yuqi/Datasets/DJI/origin/Yingrenshi_fine-registered.txt', type=str)

    
    return parser.parse_args()



def process_point(point):
    if not point:
        return -1
    else:
        most_common_label = Counter(point).most_common(1)[0][0]
        entropy = calculate_entropy(point)
        point = np.array([label for label in point if label != 0])
        label_count = len(point)
        return most_common_label, entropy, label_count




def hello(hparams: Namespace) -> None:
    hparams.ray_altitude_range = [-95, 54]
    hparams.dataset_type='memory_depth_dji'
    device = 'cpu'
    threshold=0.015

    # hparams.label_name = 'm2f' # ['m2f', 'merge', 'gt']
    runner = Runner(hparams)


    used_files = []
    for ext in ('*.png', '*.jpg'):
        used_files.extend(glob(os.path.join(hparams.far_project_m2f_paths, ext)))
        # used_files.extend(glob(os.path.join('logs_dji/augument/1015_far0.5/project_to_ori_gt/project_far_to_ori', ext)))
    used_files.sort()
    process_item = [Path(far_p).stem for far_p in used_files]



    output_path = hparams.output_path
    if not os.path.exists(output_path):
        Path(output_path).mkdir(parents=True)
    if not os.path.exists(os.path.join(output_path, 'vis_gy')):
        Path(os.path.join(output_path, 'vis_gy')).mkdir(parents=True)



    # #1. txt文件
    load_ply_path = hparams.load_ply_path
    points_nerf = np.genfromtxt(load_ply_path, usecols=(0, 1, 2))
    print(points_nerf.shape)
    root = ET.parse(hparams.metaXml_path).getroot()
    translation = np.array(root.find('SRSOrigin').text.split(',')).astype(np.float64) 
    coordinate_info = torch.load(hparams.dataset_path + '/coordinates.pt')
    origin_drb = coordinate_info['origin_drb'].numpy()
    pose_scale_factor = coordinate_info['pose_scale_factor']
    ZYQ = torch.DoubleTensor([[0, 0, -1],
                            [0, 1, 0],
                            [1, 0, 0]])
    ZYQ_1 = torch.DoubleTensor([[1, 0, 0],
                            [0, math.cos(rad(135)), math.sin(rad(135))],
                            [0, -math.sin(rad(135)), math.cos(rad(135))]])      
    points_nerf = np.array(points_nerf)
    # points_nerf = points_nerf[::100]
    print(points_nerf.shape)
    points_nerf += translation
    points_nerf = ZYQ.numpy() @ points_nerf.T
    points_nerf = (ZYQ_1.numpy() @ points_nerf).T
    points_nerf = (points_nerf - origin_drb) / pose_scale_factor

    # if not os.path.exists(f"{output_path}/ori_color_nerfcoor.ply"):   
    #     points_color = np.genfromtxt(load_ply_path, usecols=(3, 4, 5, 6))
    #     points_color = np.array(points_color)
    #     print(f"color: {points_color.shape}")

    #     cloud = PyntCloud(pd.DataFrame(
    #         # same arguments that you are passing to visualize_pcl
    #         data=np.hstack((points_nerf[:, :3], np.uint8(points_color))),
    #         columns=["x", "y", "z", "red", "green", "blue", "label"]))
    #     cloud.to_file(f"{output_path}/ori_color_nerfcoor.ply")

    # # 2. ply 文件
    # point_cloud = o3d.io.read_point_cloud("zyq/2d-3d-2d_yingrenshi_m2f/point_cloud_50.ply")
    # points_nerf = np.asarray(point_cloud.points)



    print(f"{points_nerf.shape}")




    points_nerf = torch.from_numpy(points_nerf).to(device)
    points_nerf = points_nerf.float()

    

    train_items = runner.train_items
    # train_items = train_items[:10]
    print(f"len train_items: {len(train_items)}")
    split_list=[]
    point_label_dict = {}
    point_label_list = []
    for i in range(points_nerf.shape[0]):
        point_label_list.append([])
    for metadata_item in tqdm(train_items, desc="projection"):
        
        file_name = Path(metadata_item.image_path).stem
        if file_name not in process_item:
            continue

        # if hparams.label_name == 'gt':
        #     gt_label = metadata_item.load_gt()
        # else:
        #     gt_label = metadata_item.load_label().to(device)

        #### NOTE: 这里把 m2f label 替换成想要的
        gt_label = Image.open(os.path.join(hparams.far_project_m2f_paths, f"{file_name}.png"))    #.convert('RGB')
        gt_label = torch.ByteTensor(np.asarray(gt_label))



        gt_label = remapping(gt_label)
        
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
        image = torch.zeros((image_height, image_width)).long().to(device)
        image_color = torch.zeros((image_height, image_width), dtype=torch.uint8).to(device)

        depth_map = large_int * torch.ones((image_height, image_width, 1), dtype=torch.uint8).to(device)
        depth_map_expand = large_int * torch.ones((image_height, image_width, 1), dtype=torch.uint8).to(device)

        # 获得落在图像上的点
        mask_x = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < image_width)
        mask_y = (projected_points[:, 1] >= 0) & (projected_points[:, 1] < image_height)
        mask = mask_x & mask_y

        meshMask = metadata_item.load_depth_dji().float().to(device)
        
        
        x = projected_points[:, 0].long()
        y = projected_points[:, 1].long()
        x[~mask] = 0
        y[~mask] = 0
        mesh_depths = meshMask[y, x]
        mesh_depths[~mask] = -1e6

        depth_z = pt_3d_trans[2]
        mask_z = depth_z < (mesh_depths[:, 0] + threshold)
        mask_xyz = mask & mask_z

        x, y = x[mask_xyz], y[mask_xyz]
        depth = depth_z[mask_xyz]
        idx = mask_xyz.nonzero().squeeze(-1)
        depth_map[y, x] = depth[:, None]
        image[y, x] = idx
        image_color[y, x] = gt_label[y, x]

        image_color = custom2rgb(image_color.cpu().numpy())
        
        Image.fromarray(image_color.astype(np.uint8)).save(os.path.join(output_path, 'vis_gy', f"{metadata_item.image_path.stem}.png"))
        
        # image_flatten = image.flatten()
        # gt_label_flatten = gt_label.flatten()
        # for idx, label in zip(image_flatten, gt_label_flatten):
        #     point_label_list[idx].append(label)

        # for h in range(H):
        #    for w in range(W):
        #        point_label_list[int(image[h, w])].append(int(gt_label[h, w]))
        
        image_flatten = image.flatten()
        gt_label_flatten = gt_label.flatten()
        image_flatten_non_zero = image_flatten[image_flatten.nonzero()]
        gt_label_flatten_non_zero = gt_label_flatten[image_flatten.nonzero()]
        
        for vote_idx, label_vote in zip(image_flatten_non_zero, gt_label_flatten_non_zero):
               point_label_list[vote_idx].append(label_vote)
        
    print('write pickle')
    # #将列表保存为二进制文件
    # with open(f'{output_path}/point_label_list_gy.pkl', 'wb') as file:
    #     pickle.dump(point_label_list, file)
        

    print("done")



    loaded_data = point_label_list
    # # with open(f'{output_path}/point_label_list_gy.pkl', 'rb') as file:
    # #     # 使用 pickle.load() 读取数据
    # #     loaded_data = pickle.load(file)

    # loaded_data = [[label for label in point if label != 0] for point in tqdm(loaded_data)]
    # ###1. label_num
    # label_counts = np.array([len(point) for point in loaded_data])



    ### 2. entropy and  3. 投票峰值    ## rebuttal 不计算entropy了
    most_common_labels = []
    entropies = []
    label_counts = []
    
    for point in tqdm(loaded_data, desc='calculate max_voting, no entropy, no counts'):
        # point = np.array([label for label in point if label != 0])
        point = [label for label in point if label != 0]

        if not point:
            # 处理空列表的情况
            most_common_labels.append(-1)
        else:
            most_common_labels.append(Counter(point).most_common(1)[0][0])
        # entropies.append(calculate_entropy(point))
        # label_counts.append(len(point))


    ####rebuttal  也不计数了
    # print('convert to numpy array...')
    # label_counts = np.array(label_counts)
    # print(f'label_counts: {label_counts.shape}')
    # print(f"label_counts max :{max(label_counts)}, min :{(min(label_counts))}")





    most_common_labels = np.array(most_common_labels)
    max_label = remapping(most_common_labels)
    max_label_color = custom2rgb_1(max_label)
    print(f'max_label_color: {max_label_color.shape}')



    # entropies_intensity = np.array(entropies)
    # # np.save(f"{output_path}/entropies.npy", entropies_intensity)
    # min_entropy = min(entropies_intensity)
    # max_entropy = max(entropies_intensity)
    # normalized_intensities = (entropies_intensity - min_entropy) / (max_entropy - min_entropy) * 255
    # print(f'normalized_intensities: {normalized_intensities.shape}')
    

    # print('save pc...')
    # cloud = PyntCloud(pd.DataFrame(
    #     # same arguments that you are passing to visualize_pcl
    #     data=np.hstack((points_nerf[:, :3], np.uint8(max_label_color), normalized_intensities[:, np.newaxis], label_counts[:, np.newaxis])),
    #     columns=["x", "y", "z", "red", "green", "blue", 'entropy', 'label_num']))
    # cloud.to_file(f"{output_path}/results.ply")


    print('save pc...')
    cloud = PyntCloud(pd.DataFrame(
        # same arguments that you are passing to visualize_pcl
        data=np.hstack((points_nerf[:, :3], np.uint8(max_label_color))),
        columns=["x", "y", "z", "red", "green", "blue"]))
    cloud.to_file(f"{output_path}/results.ply")


    print('done')


if __name__ == '__main__':
   

    hello(_get_train_opts())

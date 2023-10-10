import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
sys.path.append(".")

import argparse
from plyfile import PlyData
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2
import configargparse
from pathlib import Path


import json
from PIL import Image
from PIL.ExifTags import TAGS
import exifread
import math
import torch
from torchvision.utils import make_grid
import pickle

from dji.process_dji_v8_color import euler2rotation, rad
from gp_nerf.runner_gpnerf import Runner
from plyfile import PlyData, PlyElement


# 读取PLY文件
def read_ply_file(file_path):
    ply_data = PlyData.read(file_path)

    # 获取顶点坐标和颜色数据
    vertices = ply_data['vertex']
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']
    r = vertices['red']
    g = vertices['green']
    b = vertices['blue']

    # 将数据组合成列表
    data = list(zip(x, y, z, r, g, b))

    return np.array(data)

def _get_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='/data/jxchen/dataset/dji/longhua',type=str, required=False)
    parser.add_argument('--output_path', default='/data/jxchen/dataset/dji/longhua/render_dji_ply_meshMask_m2fLabel',type=str, required=False)
    parser.add_argument('--meshMask_path', default='/data/jxchen/dataset/dji/longhua',type=str)
    # parser.add_argument('--txt_path', default='/data/jxchen/dji/Yingrenshi/dji_labeled_segmented_ply.txt',type=str)
    parser.add_argument('--ply_path', default='/data/jxchen/dataset/dji/longhua/djiPly_seg_labeled.ply', type=str)
    parser.add_argument('--metaXml_path', default='/data/yuqi/Datasets/DJI/origin/Longhua_origin/block1/terra_obj_low/metadata.xml', type=str)
    parser.add_argument('--down_scale', type=int, default=4, help='')
    parser.add_argument('--alpha_cover', default=True, action='store_false')
    parser.add_argument('--m2f_label', default=True, action='store_false')

    return parser.parse_known_args()[0]


def main(hparams):
    down_scale = hparams.down_scale
    dataset_path = Path(hparams.dataset_path)  # 
    meshMask_path = Path(hparams.meshMask_path)

    # To render rgb, we need points_rgb, poses, intrics    
    # 读取包含点云的TXT文件, 读txt比读ply快非常多
    # txt_file = hparams.txt_path
    # if txt_file=='/data/jxchen/dji/Yingrenshi/dji_labeled_segmented_ply.txt':
    #     with open(txt_file, 'r') as file:
    #         data_str = file.read()

    #     # 将字符串分割成行，并使用空格分割每行中的数据
    #     lines = data_str.split('\n')
    #     data_list = [line.split() for line in lines][:-1]  # 去掉空白行

    #     [print(line,i) for i, line in enumerate(data_list) if len(line)!=6]
    #     # 将数据列表转换为 NumPy 数组
    #     data = np.array(data_list, dtype=float)  # 假设数据是浮点数类型
    # else:
    #     data = np.loadtxt(txt_file, delimiter=' ')  # 假设数据以空格分隔
    ply_data = PlyData.read(hparams.ply_path)

    # 获取XYZ坐标和标量信息列
    x = ply_data['vertex']['x']
    y = ply_data['vertex']['y']
    z = ply_data['vertex']['z']
    red = ply_data['vertex']['red']
    green = ply_data['vertex']['green']
    blue = ply_data['vertex']['blue']

    # 将RGB颜色信息转换为NumPy数组
    colors = np.column_stack((red, green, blue))

    # 将XYZ坐标和颜色信息合并为一个NumPy数组
    data = np.column_stack((x, y, z, colors))

    # 提取XYZ坐标和标量属性
    xyz = data[:, :3]  # 前三列是XYZ坐标

    scalar_attribute = np.zeros((len(colors), 1), dtype=np.uint8)  # 初始化颜色数组
    # 定义标量到颜色的映射函数
    color_mapping = {
        (0, 0, 255): 2,   # 蓝色 (Blue), terrain,2
        (0, 255, 0): 4,   # 绿色 (Green), vegetation,4
        # 2: (128, 0, 128),  # 紫色 (Purple)
        # 3: (255, 165, 0),  # 橙色 (Orange)
        (255, 255, 0): 3,  # 黄色 (Yellow), vehicle,3
        (255, 0, 0): 1,    # 红色 (Red), building,1
    }
    for key, value in color_mapping.items():
        mask = np.all(colors == key, axis=1).reshape(-1, 1)
        scalar_attribute[mask] = value

    print(f'load {hparams.ply_path}')  

    # to process points_rgb
    coordinate_info = torch.load(hparams.dataset_path + '/coordinates.pt')
    origin_drb = coordinate_info['origin_drb'].numpy()
    pose_scale_factor = coordinate_info['pose_scale_factor']

    root = ET.parse(hparams.metaXml_path).getroot()
    translation = np.array(root.find('SRSOrigin').text.split(',')).astype(np.float)    

    #######################################
    ZYQ = torch.DoubleTensor([[0, 0, -1],
                            [0, 1, 0],
                            [1, 0, 0]])
    ZYQ_1 = torch.DoubleTensor([[1, 0, 0],
                            [0, math.cos(rad(135)), math.sin(rad(135))],
                            [0, -math.sin(rad(135)), math.cos(rad(135))]])      
    points_nerf = np.array(xyz)
    points_nerf += translation
    points_nerf = ZYQ.numpy() @ points_nerf.T
    points_nerf = (ZYQ_1.numpy() @ points_nerf).T
    points_nerf = (points_nerf - origin_drb) / pose_scale_factor
    points_homogeneous = np.hstack((points_nerf, np.ones((points_nerf.shape[0], 1))))

    save_dir = hparams.output_path
    save_dir_img = os.path.join(save_dir, "img")
    save_dir_imgEx = os.path.join(save_dir, "imgEx")
    save_dir_imgCoverThres30 = os.path.join(save_dir, "alphaCover_thres30")
    save_dir_imgCoverThres20 = os.path.join(save_dir, "alphaCover_thres20")
    save_dir_imgCoverThres15 = os.path.join(save_dir, "alphaCover_thres15")
    save_dir_imgCoverThres10 = os.path.join(save_dir, "alphaCover_thres10")
    save_dir_imgCoverList = [save_dir_imgCoverThres30, save_dir_imgCoverThres20, save_dir_imgCoverThres15, 
                            save_dir_imgCoverThres10]
    thres = [0.03, 0.02, 0.015, 0.01]
    # save_dir_imgCoverList = [save_dir_imgCoverThres20]
    # thres = [0.02]

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_img, exist_ok=True)
    os.makedirs(save_dir_imgEx, exist_ok=True)
    os.makedirs(save_dir_imgCoverThres30, exist_ok=True)
    os.makedirs(save_dir_imgCoverThres20, exist_ok=True)
    os.makedirs(save_dir_imgCoverThres10, exist_ok=True)
    os.makedirs(save_dir_imgCoverThres15, exist_ok=True)
    
    alpha = 0.5
    for split in ("val", "train"):
        metadata_dir = dataset_path / split / 'metadata'
        metadata_list_dir = [os.path.join(metadata_dir, x) for x in os.listdir(metadata_dir)]
        metadata_list_dir = sorted(metadata_list_dir, key=lambda x: x[-9:-3])
        rgb_dir = dataset_path / split / 'rgbs'
        rgb_list_dir = [os.path.join(rgb_dir, x) for x in os.listdir(rgb_dir)]
        rgb_list_dir = sorted(rgb_list_dir, key=lambda x: x[-9:-3])
        meshMask_dir = meshMask_path / split/ 'depth_mesh'
        meshMask_dir = [os.path.join(meshMask_dir, x) for x in os.listdir(meshMask_dir)]
        meshMask_dir = sorted(meshMask_dir, key=lambda x: x[-10:-4])
        for metadata_file_dir, rgb_file_dir, meshMask_file_dir in tqdm(zip(metadata_list_dir, rgb_list_dir, meshMask_dir), \
            total=len(metadata_list_dir), desc=f"processing {split} set"):
            # 相机的姿态信息（相机的位置和旋转矩阵）
            metadata = torch.load(metadata_file_dir, map_location='cpu')
            camera_rotation = metadata['c2w'][:3,:3]
            camera_position = metadata['c2w'][:3, 3]
            camera_matrix = np.array([[metadata['intrinsics'][0]/down_scale, 0, metadata['intrinsics'][2]/down_scale],
                                    [0, metadata['intrinsics'][1]/down_scale, metadata['intrinsics'][3]/down_scale],
                                    [0, 0, 1]])
            # NOTE: 2. 自己写，正确
            E2 = np.hstack((camera_rotation, camera_position[:, np.newaxis]))
            E2 = np.stack([E2[:, 0], E2[:, 1]*-1, E2[:, 2]*-1, E2[:, 3]], 1)  # try
            w2c = np.linalg.inv(np.concatenate((E2, [[0,0,0,1]]), 0))
            pt_3d_trans = np.dot(w2c, points_homogeneous.T)
            pt_2d_trans = np.dot(camera_matrix, pt_3d_trans[:3]) 
            pt_2d_trans = pt_2d_trans / pt_2d_trans[2]
            projected_points = (pt_2d_trans)[:2, :]

            # 创建空白图像
            large_int = 1e6
            image_width, image_height = int(metadata['W'] / down_scale), int(metadata['H'] / down_scale)

            depth_map = large_int * np.ones((image_height, image_width, 1), dtype=np.uint8)
            depth_map_expand = large_int * np.ones((image_height, image_width, 1), dtype=np.uint8)

            npdata = 0 * np.ones((image_height, image_width, 1), dtype=np.uint8)
            image = 0 * np.ones((image_height, image_width, 3), dtype=np.uint8)
            image_expand = 0 * np.ones((image_height, image_width, 3), dtype=np.uint8)

            step = 3
            expand = True
            filename = os.path.basename(metadata_file_dir)

            mask_x = np.logical_and(projected_points[0, :] >= 0, projected_points[0, :] <= image_width)
            mask_y = np.logical_and(projected_points[1, :] >= 0, projected_points[1, :] <= image_height)
            mask = np.logical_and(mask_x, mask_y)
            # 获得深度在meshMask之内的点，用mesh深度过滤点云
            meshMask = np.load(meshMask_file_dir)
            for threshold, save_dir_imgCoverThres in zip(thres, save_dir_imgCoverList):
                # 获得落在图像上的点
                img_points = projected_points[:, mask]
                pt_3d_trans_z = pt_3d_trans[2, mask]
                depth_z = pt_3d_trans_z
                sclars = scalar_attribute[mask, :]
                rgbs = colors[mask, :]     
                image_thres = np.copy(npdata)
                image_rgb = np.copy(image)
                depth_map_thres = np.copy(depth_map)
                # image_expand_thres = np.copy(image_expand)
                for x, y, depth, scalar, color in zip(img_points[0], img_points[1], depth_z, sclars, rgbs):
                    x, y = int(x), int(y)
                    thresh = meshMask[y, x] + threshold
                
                    if depth < thresh and depth < depth_map_thres[y, x]:
                        depth_map_thres[y, x] = depth
                        cv2.circle(image_thres, (x, y), 2, (float(scalar[0])), -1)
                        cv2.circle(image_rgb, (x, y), 2, (float(color[2]), float(color[1]), float(color[0])), -1)
                        
                    # if expand and depth < depth_map_expand[y, x]:
                    #     depth_map_expand[max(0, y - step):min(image_height, y + step), max(0, x - step):min(image_width, x + step)] = depth
                    #     cv2.circle(image_expand_thres, (x, y), 2, (float(color[2]), float(color[1]), float(color[0])), -1)
                
                # depth_filter = (abs(depth_map-depth_map_expand)>(2/pose_scale_factor))
                # depth_map2 = depth_map.copy()
                # depth_map2[depth_filter]=large_int
                
                Image.fromarray(image_thres[...,0].astype(np.uint16)).save(os.path.join(save_dir_img, filename[:-3] + '.png'))
                # cv2.imwrite(os.path.join(save_dir_imgEx, filename[:-3] + '.jpg'), image_expand)

                if hparams.alpha_cover:
                    original_rgb_img = cv2.imread(rgb_file_dir)
                    new_H, new_W = original_rgb_img.shape[0]//down_scale, original_rgb_img.shape[1]//down_scale
                    original_rgb_img = cv2.resize(original_rgb_img, (new_W, new_H))

                    alpha_color_img = image_rgb * alpha + original_rgb_img * (1-alpha)
                    cv2.imwrite(os.path.join(save_dir_imgCoverThres, filename[:-3] + '.jpg'), alpha_color_img)

            

if __name__ == '__main__':
    main(_get_opts())

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
from dji.process_dji_v8_color import euler2rotation, rad
from torchvision.utils import make_grid
import os
import pickle

def visualize_scalars(scalar_tensor: torch.Tensor) -> np.ndarray:
        to_use = scalar_tensor.view(-1)
        while to_use.shape[0] > 2 ** 24:
            to_use = to_use[::2]

        mi = torch.quantile(to_use, 0.05)
        ma = torch.quantile(to_use, 0.95)

        scalar_tensor = (scalar_tensor - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
        scalar_tensor = scalar_tensor.clamp_(0, 1)

        scalar_tensor = ((1 - scalar_tensor) * 255).byte().numpy()  # inverse heatmap
        return cv2.cvtColor(cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

def _get_opts():

    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--original_images_path', default='/data/yuqi/Datasets/DJI/origin/DJI_20230726_xiayuan_data/original_image',type=str, required=False)
    parser.add_argument('--original_images_list_json_path', default='/data/yuqi/Datasets/DJI/origin/DJI_20230726_xiayuan_data/original_image_list.json',type=str, required=False)
    parser.add_argument('--infos_path', default='/data/yuqi/Datasets/DJI/origin/DJI_20230726_xiayuan_data/BlocksExchangeUndistortAT.xml',type=str, required=False)
    parser.add_argument('--output_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan',type=str, required=False)
    parser.add_argument('--resume', default=True, action='store_false')  # debug
    # parser.add_argument('--resume', default=False, action='store_true')  # run
    parser.add_argument('--num_val', type=int, default=20, help='Number of images to hold out in validation set')
    parser.add_argument('--down_scale', type=int, default=4, help='')
    return parser.parse_known_args()[0]

def main(hparams):

    down_scale = hparams.down_scale

    output_path = Path(hparams.output_path)
    original_images_path = Path(hparams.original_images_path)

    # load the pose and image name from .xml file
    root = ET.parse(hparams.infos_path).getroot()
    xml_pose = np.array([[ float(pose.find('Center/x').text),
                        float(pose.find('Center/y').text),
                        float(pose.find('Center/z').text),
                        float(pose.find('Rotation/Omega').text),
                        float(pose.find('Rotation/Phi').text),
                        float(pose.find('Rotation/Kappa').text)] for pose in root.findall('Block/Photogroups/Photogroup/Photo/Pose')])
    images_name = [Images_path.text.split("\\")[-1] for Images_path in root.findall('Block/Photogroups/Photogroup/Photo/ImagePath')]

    # load "original_images_list.json"
    with open(hparams.original_images_list_json_path, "r") as file:
        json_data = json.load(file)

    sorted_json_data = sorted(json_data, key=lambda x: x["origin_path"])  # 虽然json_data中的顺序也是按飞行顺序来，还是做个sorted保险一些
    sorted_process_data = []
    for i in tqdm(range(len(sorted_json_data)), desc="load meta_data"):
        sorted_process_line = {}
        json_line = json_data[i]
        id = json_line['id']
        index = images_name.index(id+'.jpg')
        sorted_process_line['images_name'] = images_name[index]
        
        path_segments = json_line['origin_path'].split('/')
        last_two_path = '/'.join(path_segments[-2:])
        sorted_process_line['original_image_name'] = last_two_path
        sorted_process_line['pose'] = xml_pose[index, :]
        sorted_process_data.append(sorted_process_line)

    xml_pose_sorted = np.array([x["pose"] for x in sorted_process_data])
    images_name_sorted = [x["images_name"] for x in sorted_process_data]

    # process the poses info to the requirement of nerf
    pose_dji = xml_pose_sorted
    camera_positions = pose_dji[:,0:3]#.astype(np.float32)
    camera_rotations = pose_dji[:, 3:6]#.astype(np.float32)
    camera = np.array([float(root.findall('Block/Photogroups/Photogroup/FocalLengthPixels')[0].text),
                        float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/x')[0].text),
                        float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/y')[0].text)])
    aspect_ratio = float(root.findall('Block/Photogroups/Photogroup/AspectRatio')[0].text)
    camera_matrix = np.array([[camera[0], 0, camera[1]],
                                [0, camera[0]*aspect_ratio, camera[2]],
                                [0, 0, 1]])

    
    #######-------------以上是dji/process_dji_v8_color.py中的内容，对pose和img进行了排序和转换-------------#########

    # if not os.path.exists(hparams.output_path):
    #     output_path.mkdir(parents=True)
    # (output_path / 'train' / 'depth_dji').mkdir(parents=True, exist_ok=True)
    # (output_path / 'val' / 'depth_dji').mkdir(parents=True, exist_ok=True)


    # 读取点云数据和颜色信息
    points = np.load('dji/project_script/ply2_Mmerge/points.npy')
    colors = np.load('dji/project_script/ply2_Mmerge/colors.npy')

    # scale 变化
    coordinate_info = torch.load('/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/coordinates.pt')
    origin_drb = coordinate_info['origin_drb'].numpy()
    pose_scale_factor = coordinate_info['pose_scale_factor']

    with open('dji/project_script/ply/points_lidar_list.pkl', "rb") as file:
        points_lidar_list = pickle.load(file)
        
    for i, rgb_name in enumerate(tqdm(images_name_sorted)):
        if i %50 !=0:
        # if i <= 1750:
        # if i <= 1500 or i > 1750 :
            continue
        if i % int(camera_positions.shape[0] / hparams.num_val) == 0:
            split_dir = output_path / 'val'
        else:
            split_dir = output_path / 'train'


        metadata = torch.load(str(split_dir) + '/metadata/{0:06d}.pt'.format(i), map_location='cpu')
        

        #######################################
        ZYQ = torch.DoubleTensor([[0, 0, -1],
                                [0, 1, 0],
                                [1, 0, 0]])
        ZYQ_1 = torch.DoubleTensor([[1, 0, 0],
                                [0, math.cos(rad(135)), math.sin(rad(135))],
                                [0, -math.sin(rad(135)), math.cos(rad(135))]])      
        points_nerf = points_lidar_list[i][:,:3] 
        if i - 1 >= 0:
            points_nerf = np.concatenate((points_nerf, points_lidar_list[i-1][:,:3]), axis=0)
        if i + 1 <= len(images_name_sorted):
            points_nerf = np.concatenate((points_nerf, points_lidar_list[i+1][:,:3]), axis=0)

        points_nerf = np.array(points_nerf)
        # points_nerf = np.hstack((points_nerf, np.ones((points_nerf.shape[0], 1))))
        # points_nerf = (points_nerf @ P)[:,:3]
        points_nerf = ZYQ.numpy() @ points_nerf.T
        points_nerf = (ZYQ_1.numpy() @ points_nerf).T

        points_nerf = (points_nerf - origin_drb) / pose_scale_factor

        # 相机的姿态信息（相机的位置和旋转矩阵）
        camera_rotation = metadata['c2w'][:3,:3]
        camera_position = metadata['c2w'][:3, 3]
        camera_matrix = np.array([[metadata['intrinsics'][0]/down_scale, 0, metadata['intrinsics'][2]/down_scale],
                                [0, metadata['intrinsics'][1]/down_scale, metadata['intrinsics'][3]/down_scale],
                                [0, 0, 1]])
        # NOTE: 2. 自己写，正确
        E2 = np.hstack((camera_rotation, camera_position[:, np.newaxis]))
        E2 = np.stack([E2[:, 0], E2[:, 1]*-1, E2[:, 2]*-1, E2[:, 3]], 1)
        w2c = np.linalg.inv(np.concatenate((E2, [[0,0,0,1]]), 0))
        points_homogeneous = np.hstack((points_nerf, np.ones((points_nerf.shape[0], 1))))
        pt_3d_trans = np.dot(w2c, points_homogeneous.T)

        pt_2d_trans = np.dot(camera_matrix, pt_3d_trans[:3]) 
        pt_2d_trans = pt_2d_trans / pt_2d_trans[2]
        projected_points = (pt_2d_trans)[:2, :]

        # 创建空白图像
        image_width, image_height = int(5472 / down_scale), int(3648 / down_scale)
        depth_map = 1e6 * np.ones((image_height, image_width, 1), dtype=np.uint8)
        # depth_map = np.full((image_height, image_width, 1), 1e6, dtype=np.uint8)


        # 绘制投影点到图像上（根据颜色信息进行着色）
        # mask_x1 = projected_points[0, :]>=0
        # mask_x2 = projected_points[0, :]<=image_width
        # mask_y1 = projected_points[1, :]>=0
        # mask_y2 = projected_points[1, :]<=image_height
        # mask = mask_x1 * mask_x2 * mask_y1 * mask_y2
        mask_x = np.logical_and(projected_points[0, :] >= 0, projected_points[0, :] <= image_width)
        mask_y = np.logical_and(projected_points[1, :] >= 0, projected_points[1, :] <= image_height)
        mask = np.logical_and(mask_x, mask_y)

        projected_points = projected_points[:, mask]
        pt_3d_trans_z = pt_3d_trans[2, mask]

        # 表示的是三维点到相机图像平面的距离, 不考虑相机的朝向
        depth_z = pt_3d_trans_z
        # 考虑相机朝向， 则用 三维投影点的（x,y）-> 相机中心的距离
        # depth_z = 
        
        for x, y, depth in tqdm(zip(projected_points[0], projected_points[1], depth_z)):
            x, y = int(x), int(y)
            if depth < depth_map[y, x]:
                step = 0
                if step != 0: 
                    depth_map[max(0, y - step):min(image_height, y + step), max(0, x - step):min(image_width, x + step)] = depth
                else:
                    depth_map[y, x] = depth

        # for j in tqdm(range(projected_points.shape[1])):
        #     x, y = projected_points[:,j]
        #     x, y = int(x), int(y)
        #     if depth_z[j] < depth_map[y, x]:
        #         step = 2
        #         depth_map[max(0, y-step):min(image_height, y+step), max(0, x-step):min(image_width, x+step)] = depth_z[j]
        # np.save(split_dir / 'depth_dji' / '{0:06d}.npy'.format(i), depth_map)
            

        depth_mask = (depth_map!=1e6)
        
        depth_map_valid = depth_map[depth_mask]
        min_depth = depth_map_valid.min()
        max_depth = depth_map_valid.max()

        depth_vis1 = depth_map
        depth_vis1[depth_map==1e6] =  min_depth 

        img_list1=[]
        

        depth_vis1 = (depth_vis1 - min_depth) / max(max_depth - min_depth, 1e-8)  # normalize to 0~1
        depth_vis1 = torch.from_numpy(depth_vis1).clamp_(0, 1)
        depth_vis1 = ((1 - depth_vis1) * 255).byte().numpy()  # inverse heatmap
        depth_vis1 = cv2.cvtColor(cv2.applyColorMap(depth_vis1, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        
        # depth_vis1 = visualize_scalars(torch.log(torch.from_numpy(depth_vis1) + 1e-8).cpu())
        
        img_list1.append(torch.from_numpy(depth_vis1))
        img_list1 = torch.stack(img_list1).permute(0,3,1,2)
        img = make_grid(img_list1, nrow=3)
        img_grid1 = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        img_grid1[~depth_mask[:,:,0]] = [255,255,255]
        Image.fromarray(img_grid1).save('dji/project_script/ply/output/'+'{0:06d}_3_depth.jpg'.format(i))
        
    a = 1 

if __name__ == '__main__':
    main(_get_opts())
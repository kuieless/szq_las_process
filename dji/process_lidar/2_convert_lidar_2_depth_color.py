
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

from dji.opt import _get_opts


def main(hparams):
    print(f'hparams.debug: {hparams.debug}')
    down_scale = hparams.down_scale

    dataset_path = Path(hparams.dataset_path)
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
        if (id+'.jpg') in images_name:
            index = images_name.index(id+'.jpg')
        else:
            continue
        # index = images_name.index(id+'.jpg')
        sorted_process_line['images_name'] = images_name[index]
        
        path_segments = json_line['origin_path'].split('/')
        last_two_path = '/'.join(path_segments[-2:])
        sorted_process_line['original_image_name'] = last_two_path
        sorted_process_line['pose'] = xml_pose[index, :]
        sorted_process_data.append(sorted_process_line)

    xml_pose_sorted = np.array([x["pose"] for x in sorted_process_data])
    images_name_sorted = [x["images_name"] for x in sorted_process_data]
    original_image_name_sorted = [x["original_image_name"] for x in sorted_process_data]


    # process the poses info to the requirement of nerf
    pose_dji = xml_pose_sorted
    camera_positions = pose_dji[:,0:3]#.astype(np.float32)
    camera_rotations = pose_dji[:, 3:6]#.astype(np.float32)
    camera = np.array([float(root.findall('Block/Photogroups/Photogroup/FocalLengthPixels')[0].text),
                        float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/x')[0].text),
                        float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/y')[0].text)])
    aspect_ratio = float(root.findall('Block/Photogroups/Photogroup/AspectRatio')[0].text)
    camera_matrix_ori = np.array([[camera[0], 0, camera[1]],
                                [0, camera[0]*aspect_ratio, camera[2]],
                                [0, 0, 1]])


    # distortion_coeffs1 = np.array([float(-0.009831534),
    #                        float(0.013398247),
    #                        float(-0.002080232),
    #                        float(-0.001410897),
    #                        float(-0.010848353)])  # k1 k2 p1 p2 k3
    # camera_matrix1 = np.array([[3695.607, 0, 2713.97],
    #                         [0, 3695.607, 1811.31],
    #                         [0, 0, 1]])
    
    distortion_coeffs1 = np.array([float(hparams.k1),
                           float(hparams.k2),
                           float(hparams.p1),
                           float(hparams.p2),
                           float(hparams.k3)])  # k1 k2 p1 p2 k3
    camera_matrix1 = np.array([[hparams.fx, 0, hparams.cx],
                            [0, hparams.fx, hparams.cy],
                            [0, 0, 1]])
    
    #######-------------以上是dji/process_dji_v8_color.py中的内容，对pose和img进行了排序和转换-------------#########

    

    # scale 变化
    coordinate_info = torch.load(hparams.dataset_path + '/coordinates.pt')
    origin_drb = coordinate_info['origin_drb'].numpy()
    pose_scale_factor = coordinate_info['pose_scale_factor']

    with open(hparams.las_output_path+'/points_lidar_list.pkl', "rb") as file:
        points_lidar_list = pickle.load(file)
    print(f'load {hparams.las_output_path}')
    
    if hparams.debug:
        output_path = Path(hparams.output_path)
        if not os.path.exists(hparams.output_path):
            output_path.mkdir(parents=True)
        (output_path / 'debug').mkdir(parents=True, exist_ok=True)
    else:
        if not os.path.exists(str(dataset_path / 'val' / 'depth_dji')):
            (dataset_path / 'val' / 'depth_dji').mkdir()
        if not os.path.exists(str(dataset_path / 'train' / 'depth_dji')):
            (dataset_path / 'train' / 'depth_dji').mkdir()


    for i, rgb_name in enumerate(tqdm(images_name_sorted)):
        ## 确保在las文件范围内, 比如有五条航线，每个航线有对应的数量
        if i < hparams.start or i >= hparams.end:
            continue
        
        # #### 最后几张图的结果
        # if i < (hparams.end-3):
        #     continue
        
        # 调试用，输出前几张和后几张，以及中间几张的结果
        if hparams.debug:
            if i % 50 !=0:
                if i > (hparams.end-4) or i < (hparams.start+2):
                    pass
                else:
                    continue
            

        if i % int(camera_positions.shape[0] / hparams.num_val) == 0:
            split_dir = dataset_path / 'val'
        else:
            continue
            split_dir = dataset_path / 'train'
        

        img1 = cv2.imread(str(original_images_path) + '/' + original_image_name_sorted[i]) 
        img_change = cv2.undistort(img1, camera_matrix1, distortion_coeffs1, None, camera_matrix_ori)
        img_change = cv2.resize(img_change, (int(img_change.shape[1]/hparams.down_scale), int(img_change.shape[0]/hparams.down_scale)))
        current_i = i - hparams.start
        if hparams.debug:
            cv2.imwrite(str(output_path / 'debug' / '{0:06d}_1_rgbs.png'.format(i)), img_change)
            
            
            if current_i < (hparams.end - hparams.start - 2):
                points_color = points_lidar_list[current_i][:,3:] 
                #### 获取前后两帧的颜色
                if current_i - 1 >= 0:
                    points_color = np.concatenate((points_color, points_lidar_list[current_i-1][:,3:]), axis=0)
                if current_i + 1 < (hparams.end - hparams.start):
                    points_color = np.concatenate((points_color, points_lidar_list[current_i+1][:,3:]), axis=0)
            else:
                points_color = np.concatenate((points_lidar_list[-1][:,3:], points_lidar_list[-2][:,3:], points_lidar_list[-3][:,3:]), axis=0)
                
        # ####获取前后两帧的点云
        if current_i < (hparams.end - hparams.start - 2):
            points_nerf = points_lidar_list[current_i][:,:3] 
            if current_i - 1 >= 0:
                points_nerf = np.concatenate((points_nerf, points_lidar_list[current_i-1][:,:3]), axis=0)
            if current_i + 1 < (hparams.end - hparams.start):
                points_nerf = np.concatenate((points_nerf, points_lidar_list[current_i+1][:,:3]), axis=0)
        else:
            points_nerf = np.concatenate((points_lidar_list[-1][:,:3], points_lidar_list[-2][:,:3], points_lidar_list[-3][:,:3]), axis=0)

        
        
        #######################################
        ZYQ = torch.DoubleTensor([[0, 0, -1],
                                [0, 1, 0],
                                [1, 0, 0]])
        ZYQ_1 = torch.DoubleTensor([[1, 0, 0],
                                [0, math.cos(rad(135)), math.sin(rad(135))],
                                [0, -math.sin(rad(135)), math.cos(rad(135))]])      
        
        
        
        


        points_nerf = np.array(points_nerf)
        points_nerf = ZYQ.numpy() @ points_nerf.T
        points_nerf = (ZYQ_1.numpy() @ points_nerf).T

        points_nerf = (points_nerf - origin_drb) / pose_scale_factor

        # 相机的姿态信息（相机的位置和旋转矩阵）
        metadata = torch.load(str(split_dir) + '/metadata/{0:06d}.pt'.format(i), map_location='cpu')
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
        large_int = 1e6
        image_width, image_height = int(5472 / down_scale), int(3648 / down_scale)
        depth_map = large_int * np.ones((image_height, image_width, 1), dtype=np.uint8)
        depth_map_expand = large_int * np.ones((image_height, image_width, 1), dtype=np.uint8)

        image = 255 * np.ones((image_height, image_width, 3), dtype=np.uint8)
        image_expand = 255 * np.ones((image_height, image_width, 3), dtype=np.uint8)


        # 获得落在图像上的点
        mask_x = np.logical_and(projected_points[0, :] >= 0, projected_points[0, :] <= image_width)
        mask_y = np.logical_and(projected_points[1, :] >= 0, projected_points[1, :] <= image_height)
        mask = np.logical_and(mask_x, mask_y)

        projected_points = projected_points[:, mask]
        pt_3d_trans_z = pt_3d_trans[2, mask]
        depth_z = pt_3d_trans_z
        
        step = 3
        expand = True
        if hparams.debug == False: # 投影深度和颜色
            # 只投影深度, 并保存在数据文件夹下
            for x, y, depth in tqdm(zip(projected_points[0], projected_points[1], depth_z)):
                x, y = int(x), int(y)
                if depth < depth_map[y, x]:
                    depth_map[y, x] = depth
                if expand and depth < depth_map_expand[y, x]:
                    depth_map_expand[max(0, y - step):min(image_height, y + step), max(0, x - step):min(image_width, x + step)] = depth
            
            depth_filter = (abs(depth_map-depth_map_expand)>(2/pose_scale_factor))
            depth_map2 = depth_map.copy()
            depth_map2[depth_filter]=large_int

            ## 以下是可视化代码
            # img_list2=[]
            # invalid_mask2 = (depth_map2==large_int)
            # depth_map_valid2 = depth_map2[~invalid_mask2]
            # min_depth2 = depth_map_valid2.min()
            # max_depth2 = depth_map_valid2.max()
            # depth_vis2 = depth_map2
            # depth_vis2[invalid_mask2] =  min_depth2 
            # depth_vis2 = (depth_vis2 - min_depth2) / max(max_depth2 - min_depth2, 1e-8)  # normalize to 0~1
            # depth_vis2 = torch.from_numpy(depth_vis2).clamp_(0, 1)
            # depth_vis2 = ((1 - depth_vis2) * 255).byte().numpy()  # inverse heatmap
            # depth_vis2 = cv2.cvtColor(cv2.applyColorMap(depth_vis2, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        
            # img_list2.append(torch.from_numpy(depth_vis2))
            # img_list2 = torch.stack(img_list2).permute(0,3,1,2)
            # img2 = make_grid(img_list2, nrow=3)
            # img_grid2 = img2.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            # img_grid2[invalid_mask2[:,:,0]] = [255,255,255]
            # Image.fromarray(img_grid2).save(f'{hparams.output_path}/debug/{i:06d}_3_depth_filter.png')
            
            np.save(split_dir / 'depth_dji' / '{0:06d}.npy'.format(i), depth_map2)

        else:
            points_color_mask = points_color[mask]
            for x, y, depth, color in tqdm(zip(projected_points[0], projected_points[1], depth_z, points_color_mask)):
                x, y = int(x), int(y)
                if depth < depth_map[y, x]:
                    depth_map[y, x] = depth
                    image[y, x] = color[::-1]
                    cv2.circle(image_expand, (x, y), 2, (float(color[2]), float(color[1]), float(color[0])), -1)
                if expand and depth < depth_map_expand[y, x]:
                    depth_map_expand[max(0, y - step):min(image_height, y + step), max(0, x - step):min(image_width, x + step)] = depth
            # cv2.imwrite(str(output_path / 'debug' / '{0:06d}_2_project.png'.format(i)), image)
            if expand:
                # cv2.imwrite(str(output_path / 'debug' / '{0:06d}_2_project_size3.png'.format(i)), image_expand)


                #####  膨胀的深度图  ############################ ############################
                # img_list3=[]
                # invalid_mask3 = (depth_map_expand==large_int)
                # depth_map_valid3 = depth_map_expand[~invalid_mask3]
                # min_depth3 = depth_map_valid3.min()
                # max_depth3 = depth_map_valid3.max()
                # depth_vis3 = depth_map_expand
                # depth_vis3[invalid_mask3] =  min_depth3
                # depth_vis3 = (depth_vis3 - min_depth3) / max(max_depth3 - min_depth3, 1e-8)  # normalize to 0~1
                # depth_vis3 = torch.from_numpy(depth_vis3).clamp_(0, 1)
                # depth_vis3 = ((1 - depth_vis3) * 255).byte().numpy()  # inverse heatmap
                # depth_vis3 = cv2.cvtColor(cv2.applyColorMap(depth_vis3, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                # img_list3.append(torch.from_numpy(depth_vis3))
                # img_list3 = torch.stack(img_list3).permute(0,3,1,2)
                # img3 = make_grid(img_list3, nrow=3)
                # img_grid3 = img3.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                # img_grid3[invalid_mask3[:,:,0]] = [255,255,255]
                # Image.fromarray(img_grid3).save(str(output_path / 'debug' / '{0:06d}_3_depth_expand.png').format(i))


                #####  过滤的深度图 ############################ ############################
                depth_filter = (abs(depth_map-depth_map_expand)>(2/pose_scale_factor))
                depth_map2 = depth_map.copy()
                depth_map2[depth_filter]=large_int
                img_list2=[]
                invalid_mask2 = (depth_map2==large_int)
                depth_map_valid2 = depth_map2[~invalid_mask2]
                min_depth2 = depth_map_valid2.min()
                max_depth2 = depth_map_valid2.max()
                depth_vis2 = depth_map2
                depth_vis2[invalid_mask2] =  min_depth2 
                depth_vis2 = (depth_vis2 - min_depth2) / max(max_depth2 - min_depth2, 1e-8)  # normalize to 0~1
                depth_vis2 = torch.from_numpy(depth_vis2).clamp_(0, 1)
                depth_vis2 = ((1 - depth_vis2) * 255).byte().numpy()  # inverse heatmap
                depth_vis2 = cv2.cvtColor(cv2.applyColorMap(depth_vis2, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                img_list2.append(torch.from_numpy(depth_vis2))
                img_list2 = torch.stack(img_list2).permute(0,3,1,2)
                img2 = make_grid(img_list2, nrow=3)
                img_grid2 = img2.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                img_grid2[invalid_mask2[:,:,0]] = [255,255,255]
                Image.fromarray(img_grid2).save(str(output_path / 'debug' / '{0:06d}_3_depth_filter.png').format(i))
                ############################ ############################ ############################
                # image[invalid_mask2[:,:,0]] = [255,255,255]
                # cv2.imwrite(str(output_path / 'debug' / '{0:06d}_2_project.png'.format(i)), image)


            #####  没有过滤的原始深度图   ############################ ############################
            # img_list1=[]
            # invalid_mask = (depth_map==large_int)
            # depth_map_valid = depth_map[~invalid_mask]
            # min_depth = depth_map_valid.min()
            # max_depth = depth_map_valid.max()
            # depth_vis1 = depth_map
            # depth_vis1[invalid_mask] =  min_depth 
            # depth_vis1 = (depth_vis1 - min_depth) / max(max_depth - min_depth, 1e-8)  # normalize to 0~1
            # depth_vis1 = torch.from_numpy(depth_vis1).clamp_(0, 1)
            # depth_vis1 = ((1 - depth_vis1) * 255).byte().numpy()  # inverse heatmap
            # depth_vis1 = cv2.cvtColor(cv2.applyColorMap(depth_vis1, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            # img_list1.append(torch.from_numpy(depth_vis1))
            # img_list1 = torch.stack(img_list1).permute(0,3,1,2)
            # img = make_grid(img_list1, nrow=3)
            # img_grid1 = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            # img_grid1[invalid_mask[:,:,0]] = [255,255,255]
            # Image.fromarray(img_grid1).save(str(output_path / 'debug' / '{0:06d}_3_depth.png').format(i))
             ############################ ############################ ############################

            

if __name__ == '__main__':
    main(_get_opts())
#### 注意手动更改智图报告中的相机参数（优化），
# 包括fx,fy,cx,cy,k1,k2,p1,p2,k3

from argparse import Namespace
from pathlib import Path
import numpy as np
import configargparse
import os
from tqdm import tqdm
import shutil
import torch
import trimesh
import random
import math
import cv2
from dji.visual_poses import visualize_poses, load_poses
from bs4 import BeautifulSoup

import json
from PIL import Image
from PIL.ExifTags import TAGS
import exifread

def _get_opts():

    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--AT_images_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/origin/undistort',type=str, required=False)
    parser.add_argument('--original_images_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/origin/original_image',type=str, required=False)
    parser.add_argument('--original_images_list_json_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/origin/original_image_list.json',type=str, required=False)
    parser.add_argument('--infos_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/origin/BlocksExchangeUndistortAT.xml',type=str, required=False)
    parser.add_argument('--output_path', default='./test',type=str, required=False)
    parser.add_argument('--resume', default=True, action='store_false')  # debug
    # parser.add_argument('--resume', default=False, action='store_true')  # run
    parser.add_argument('--num_val', type=int, default=20, help='Number of images to hold out in validation set')
    return parser.parse_known_args()[0]

def rad(x):
    return math.radians(x)

def euler2rotation(theta, ):
    theta = [rad(i) for i in theta]
    omega, phi, kappa = theta[0], theta[1], theta[2]
    # omega, phi, kappa = rad(180)-theta[0], rad(180)-theta[1], rad(180)-theta[2]
    # omega, phi, kappa = rad(180)-theta[0], rad(180)-theta[1], theta[2]

    R_omega = np.array([[1, 0, 0],
                       [0, math.cos(omega), -math.sin(omega)],
                       [0, math.sin(omega), math.cos(omega)]])

    R_phi = np.array([[math.cos(phi), 0, math.sin(phi)],
                        [0, 1, 0],
                        [-math.sin(phi), 0, math.cos(phi)]])

    R_kappa = np.array([[math.cos(kappa), -math.sin(kappa), 0],
                      [math.sin(kappa), math.cos(kappa), 0],
                      [0, 0, 1]])

    R3d = R_omega @ R_phi @ R_kappa
    # R3d = R_kappa @ R_phi @ R_omega
    return R3d




import xml.etree.ElementTree as ET

def main(hparams):
    # initialize
    output_path = Path(hparams.output_path)
    # Note 空三后的图像颜色比较奇怪，暂时不用
    # AT_images_path = Path(hparams.AT_images_path)
    original_images_path = Path(hparams.original_images_path)
    original_images_list_json_path = Path(hparams.original_images_list_json_path)



    # load the pose and image name from .xml file
    root = ET.parse(hparams.infos_path).getroot()
    xml_pose = np.array([[ float(pose.find('Center/x').text),
                        float(pose.find('Center/y').text),
                        float(pose.find('Center/z').text),
                        float(pose.find('Rotation/Omega').text),
                        float(pose.find('Rotation/Phi').text),
                        float(pose.find('Rotation/Kappa').text)] for pose in root.findall('Block/Photogroups/Photogroup/Photo/Pose')])
    images_name = [Images_path.text.split("\\")[-1] for Images_path in root.findall('Block/Photogroups/Photogroup/Photo/ImagePath')]
    # print(images_name)
    # load "original_images_list.json"
    with open(hparams.original_images_list_json_path, "r") as file:
        json_data = json.load(file)

    sorted_json_data = sorted(json_data, key=lambda x: x["origin_path"])  # 虽然json_data中的顺序也是按飞行顺序来，还是做个sorted保险一些
    print(len(sorted_json_data))
    sorted_process_data = []
    
    for i in tqdm(range(len(sorted_json_data)), desc="load meta_data"):
        sorted_process_line = {}
        json_line = json_data[i]
        id = json_line['id']
        index = images_name.index(id+'.jpg')
        # index = images_name.index(id)

        sorted_process_line['images_name'] = images_name[index]
        
        path_segments = json_line['origin_path'].split('/')
        last_two_path = '/'.join(path_segments[-2:])
        sorted_process_line['original_image_name'] = last_two_path
        sorted_process_line['pose'] = xml_pose[index, :]
        
        # extract other basic metadata
        # # 1. 使用exifread库读取， 可读到图像的很多metadata
        # 打开图像
        image_path = hparams.original_images_path + '/' + last_two_path
        with open(image_path, 'rb') as image_file:
            # 读取图像的Exif数据
            tags = exifread.process_file(image_file)
        tags['original_image_path'] = last_two_path
        del tags['JPEGThumbnail']
        #####输出所有Exif标签及其值
        # for tag, value in tags.items():
        #     print(f"{tag:25}: {value}")
        sorted_process_line['meta_tags'] = tags

        sorted_process_data.append(sorted_process_line)

        

    xml_pose_sorted = np.array([x["pose"] for x in sorted_process_data])
    images_name_sorted = [x["images_name"] for x in sorted_process_data]
    original_image_name_sorted = [x["original_image_name"] for x in sorted_process_data]

    # process the poses info to the requirement of nerf
    pose_dji = xml_pose_sorted
    camera_positions = pose_dji[:,0:3]#.astype(np.float32)
    camera_rotations = pose_dji[:, 3:6]#.astype(np.float32)

    c2w_R = []
    for i in range(len(camera_rotations)):
        R_temp = euler2rotation(camera_rotations[i])
        c2w_R.append(R_temp)

    ZYQ = torch.DoubleTensor([[0, 0, -1],
                             [0, 1, 0],
                             [1, 0, 0]])
    ZYQ_1 = torch.DoubleTensor([[1, 0, 0],
                              [0, math.cos(rad(135)), math.sin(rad(135))],
                              [0, -math.sin(rad(135)), math.cos(rad(135))]])

    c2w = []
    for i in range(len(c2w_R)):
        temp = np.concatenate((c2w_R[i], camera_positions[i:i + 1].T), axis=1)
        temp = np.concatenate((temp[:,0:1], -temp[:,1:2], -temp[:,2:3], temp[:,3:]), axis=1)
        temp = torch.hstack((ZYQ @ temp[:3, :3], ZYQ @ temp[:3, 3:]))
        temp = torch.hstack((ZYQ_1 @ temp[:3, :3], ZYQ_1 @ temp[:3, 3:]))
        temp = temp.numpy()
        c2w.append(temp)
    c2w = np.array(c2w)

    min_position = np.min(c2w[:,:, 3], axis=0)
    max_position = np.max(c2w[:,:, 3], axis=0)
    print('Coord range: {} {}'.format(min_position, max_position))
    origin = (max_position + min_position) * 0.5
    dist = (torch.tensor(c2w[:,:, 3]) - torch.tensor(origin)).norm(dim=-1)
    diagonal = dist.max()
    scale = diagonal.numpy()
    c2w[:,:, 3] = (c2w[:,:, 3] - origin) / scale
    assert np.logical_and(c2w[:,:, 3] >= -1, c2w[:,:, 3] <= 1).all()

    if not os.path.exists(hparams.output_path):
        output_path.mkdir(parents=True)
    (output_path / 'train' / 'image_metadata').mkdir(parents=True)
    (output_path / 'train' / 'metadata').mkdir(parents=True)
    (output_path / 'train' / 'rgbs').mkdir(parents=True)
    (output_path / 'val' / 'image_metadata').mkdir(parents=True)
    (output_path / 'val' / 'metadata').mkdir(parents=True)
    (output_path / 'val' / 'rgbs').mkdir(parents=True)

    # pose_dji = load_poses(dataset_path='/disk1/Datasets/dji/DJI-XML-colmap')
    # # 这里pose dji只有1071个，而c2w有1092个，可视化的时候要注意
    # visualize_poses(c2w[0:2], pose_dji[0:2])
    # visualize_poses

    # visualize_poses(random.sample(c2w,200))

    coordinates = {
        'origin_drb': torch.Tensor(origin),
        'pose_scale_factor': scale
    }
    torch.save(coordinates, output_path / 'coordinates.pt')

    camera = np.array([float(root.findall('Block/Photogroups/Photogroup/FocalLengthPixels')[0].text),
                       float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/x')[0].text),
                       float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/y')[0].text)])
    aspect_ratio = float(root.findall('Block/Photogroups/Photogroup/AspectRatio')[0].text)
    camera_matrix = np.array([[camera[0], 0, camera[1]],
                              [0, camera[0]*aspect_ratio, camera[2]],
                              [0, 0, 1]])
    distortion = np.array([float(root.findall('Block/Photogroups/Photogroup/Distortion/K1')[0].text),
                           float(root.findall('Block/Photogroups/Photogroup/Distortion/K2')[0].text),
                           float(root.findall('Block/Photogroups/Photogroup/Distortion/P1')[0].text),
                           float(root.findall('Block/Photogroups/Photogroup/Distortion/P2')[0].text),
                           float(root.findall('Block/Photogroups/Photogroup/Distortion/K3')[0].text)])  # k1 k2 p1 p2 k3

    with (output_path / 'mappings.txt').open('w') as f:
        for i, rgb_name in enumerate(tqdm(images_name_sorted)):
            if i % int(camera_positions.shape[0] / hparams.num_val) == 0:
                split_dir = output_path / 'val'
            else:
                split_dir = output_path / 'train'
            if True:
                # 这个是空三后的图像， 颜色会被矫正，导致颜色有点奇怪
                # rgb_ori = str(AT_images_path) + '/' + rgb_name
                # distorted = cv2.imread(rgb_ori)
                # undistorted = cv2.undistort(distorted, camera_matrix, distortion)#这里因为distortion为[0,0,0,0,0],所以等于没有做任何操作
                # cv2.imwrite(str(split_dir / 'rgbs' / '{0:06d}.jpg'.format(i)), undistorted)
                
                # NOTE 根据 dji智图质量报告上的数值进行修改
                distortion1 = np.array([float(-0.014544237),
                           float(0.004694901),
                           float(-0.001346389),
                           float(-0.00096356),
                           float(0.004985782)])  # k1 k2 p1 p2 k3
                camera_matrix1 = np.array([[3692.801, 0, 2730.325],
                                        [0, 3692.801, 1817.382],
                                        [0, 0, 1]])
                img1 = cv2.imread(str(original_images_path) + '/' + original_image_name_sorted[i]) 
                img_change = cv2.undistort(img1, camera_matrix1, distortion1, None, camera_matrix)
                cv2.imwrite(str(split_dir / 'rgbs' / '{0:06d}.jpg'.format(i)), img_change)


            # label = 'others'
            # omega, phi, kappa = xml_pose_sorted[i][3], xml_pose_sorted[i][4], xml_pose_sorted[i][5]
            # if (140 <= omega <= 150) and (25 <= phi <= 36) and (45 <= kappa <= 55):
            #     label = 'right'
            # elif (-150 <= omega <= -140) and (-36 <= phi <= -25) and (-135 <= kappa <= -125):
            #     label = 'right'
            # elif (145 <= omega <= 155) and (25 <= phi <= 36) and (-135 <= kappa <= -125):
            #     label = 'left'
            # elif (-155 <= omega <= -145) and (-36 <= phi <= -25) and (45 <= kappa <= 55):
            #     label = 'left'


            metadata_name = '{0:06d}.pt'.format(i)
            torch.save({
                'H': img1.shape[0],
                'W': img1.shape[1],
                'c2w': torch.FloatTensor(c2w[i]),
                'intrinsics': torch.FloatTensor(
                    [camera_matrix[0][0], camera_matrix[1][1], camera_matrix[0][2], camera_matrix[1][2]]),
                'distortion': torch.FloatTensor(distortion),
                # 'label': label
            }, split_dir / 'metadata' / metadata_name)

            f.write('{},{}\n'.format(rgb_name, metadata_name))


            torch.save(sorted_process_data[i], split_dir / 'image_metadata' / '{0:06d}.pt'.format(i))

    print('Finish')


if __name__ == '__main__':
    main(_get_opts())
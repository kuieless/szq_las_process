#
#  这个代码是用于处理大疆智图空三导出的xml文件
#  首先，将xml的gos坐标系转换成cecf坐标系
#  再进行缩放旋转，使其朝向+x轴，
#  ps：根据xml的yaw pitch roll赋予label，在训练中去除黑影（脚架）
#

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

from scipy.spatial.transform import Rotation

from scripts_from_jx.viz_pose_obj import visualize_poses_and_points

def ypr_to_opk(input):
    yaw, pitch, roll = input[0], input[1], input[2]
    # r = Rotation.from_euler('ZYX',[yaw,pitch,roll],degrees=True)
    # r = Rotation.from_euler('XYZ',[yaw,pitch,roll],degrees=True)
    # r = Rotation.from_euler('ZXY',[yaw,pitch,roll],degrees=True)
    # r = Rotation.from_euler('XZY',[yaw,pitch,roll],degrees=True)
    # r = Rotation.from_euler('YZX',[yaw,pitch,roll],degrees=True)
    # r = Rotation.from_euler('YXZ',[yaw,pitch,roll],degrees=True)

    opk=r.as_rotvec()
    return opk

def rad(x):
    return math.radians(x)

def euler2rotation(theta, ):
    theta = [rad(i) for i in theta]
    omega, phi, kappa = theta[2], theta[1], theta[0]
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

def ypr2rotation(input):
    input = [rad(i) for i in input]
    Y, P, R = input[0], input[1], input[2]
    
    R3d = np.array([  
        [math.cos(R) * math.cos(Y) - math.sin(R) * math.sin(P) * math.sin(Y), -math.cos(R) * math.sin(Y) - math.cos(Y) * math.sin(R) * math.sin(P), math.cos(P) * math.sin(R)],  
        [math.cos(Y) * math.sin(R) + math.cos(R) * math.sin(P) * math.sin(Y), math.cos(R) * math.cos(Y) * math.sin(P) - math.sin(R) * math.sin(Y), -math.cos(R) * math.cos(P)],  
        [math.cos(P) * math.sin(Y), math.cos(P) * math.cos(Y), math.sin(P)]  
    ])
    return R3d



def _get_opts():

    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--images_path', default='/data/yuqi/Datasets/InstanceBuilding/3D/scene1/photos',type=str, required=False)
    parser.add_argument('--infos_path', default='/data/yuqi/Datasets/InstanceBuilding/3D/scene1/photos-info-pyr.xml',type=str, required=False)
    parser.add_argument('--output_path', default='/data/yuqi/Datasets/InstanceBuilding/3D/scene1/output',type=str, required=False)
    parser.add_argument('--resume', default=True, action='store_false')  # debug
    parser.add_argument('--num_val', type=int, default=1000, help='Number of images to hold out in validation set')
    return parser.parse_known_args()[0]

import xml.etree.ElementTree as ET

def main(hparams):
    # initialize
    output_path = Path(hparams.output_path)
    images_path = Path(hparams.images_path)

    if not os.path.exists(hparams.output_path):
        output_path.mkdir(parents=True)
    (output_path / 'train' / 'metadata').mkdir(parents=True,exist_ok=True)
    (output_path / 'val' / 'metadata').mkdir(parents=True,exist_ok=True)
    (output_path / 'train' / 'rgbs').mkdir(parents=True,exist_ok=True)
    (output_path / 'val' / 'rgbs').mkdir(parents=True,exist_ok=True)

    root = ET.parse(hparams.infos_path).getroot()
    xml_pose = np.array([[ float(pose.find('Center/x').text),
                        float(pose.find('Center/y').text),
                        float(pose.find('Center/z').text),
                        float(pose.find('Rotation/Yaw').text),
                        float(pose.find('Rotation/Pitch').text),
                        float(pose.find('Rotation/Roll').text)] for pose in root.findall('Photogroup/Photo/Pose')])
    # xml_pose = np.array([[float(pose.find('Center/x').text),
    #                       float(pose.find('Center/y').text),
    #                       float(pose.find('Center/z').text),
    #                       float(pose.find('Rotation/Pitch').text),
    #                       float(pose.find('Rotation/Roll').text),
    #                       float(pose.find('Rotation/Yaw').text)] for pose in root.findall('Photogroup/Photo/Pose')])
    images_name = [Images_path.text.split("\\")[-1] for Images_path in root.findall('Photogroup/Photo/ImageName')]


    # process the poses info to the requirement of nerf
    pose_dji = xml_pose
    camera_positions = pose_dji[:,0:3]#.astype(np.float32)
    np.savetxt('scene1_camera.txt', camera_positions)
    #
    # min_position = np.min(camera_positions, axis=0)
    # max_position = np.max(camera_positions, axis=0)
    # print('Coord range: {} {}'.format(min_position, max_position))
    # origin = (max_position + min_position) * 0.5
    # dist = (torch.tensor(camera_positions) - torch.tensor(origin)).norm(dim=-1)
    # diagonal = dist.max()
    # scale = diagonal.numpy()
    # camera_positions = (camera_positions - origin) / scale
    # assert np.logical_and(camera_positions >= -1, camera_positions <= 1).all()

    camera_rotations = pose_dji[:, 3:6]#.astype(np.float32)

    c2w_R = []
    for i in range(len(camera_rotations)):
        # opk = ypr_to_opk(camera_rotations[i])
        # R_temp = euler2rotation(opk)
        R_temp = ypr2rotation(camera_rotations[i])
        c2w_R.append(R_temp)







    ZYQ = torch.DoubleTensor([[0, 0, -1],
                             [0, 1, 0],
                             [1, 0, 0]])
    ZYQ_1 = torch.DoubleTensor([[1, 0, 0],
                              [0, math.cos(rad(135)), math.sin(rad(135))],
                              [0, -math.sin(rad(135)), math.cos(rad(135))]])

    Y_180 = torch.DoubleTensor([[-1, 0, 0],
                             [0, 1, 0],
                             [0, 0, -1]])

    c2w = []
    for i in range(len(c2w_R)):
        temp = np.concatenate((c2w_R[i], camera_positions[i:i + 1].T), axis=1)
        temp = np.concatenate((temp[:,0:1], -temp[:,1:2], -temp[:,2:3], temp[:,3:]), axis=1)
        temp = torch.hstack((ZYQ @ temp[:3, :3], ZYQ @ temp[:3, 3:]))
        temp = torch.hstack((ZYQ_1 @ temp[:3, :3], ZYQ_1 @ temp[:3, 3:]))
        # temp = torch.hstack((Y_180 @ temp[:3, :3], Y_180 @ temp[:3, 3:]))


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


    # visualize_poses(c2w)
    # visualize_poses_and_points(c2w,mesh=trimesh.load_mesh('/data/yuqi/Datasets/InstanceBuilding/3D/scene1/norm.obj'))


    coordinates = {
        'origin_drb': torch.Tensor(origin),
        'pose_scale_factor': scale
    }
    torch.save(coordinates, output_path / 'coordinates.pt')

    camera = np.array([float(root.findall('Photogroup/FocalLength')[0].text),
                       float(root.findall('Photogroup/PrincipalPoint/x')[0].text),
                       float(root.findall('Photogroup/PrincipalPoint/y')[0].text)])
    Width = float(root.findall('Photogroup/ImageDimensions/Width')[0].text)
    Height = float(root.findall('Photogroup/ImageDimensions/Height')[0].text)
    FocalLength = float(root.findall('Photogroup/FocalLength')[0].text)
    SensorSize = float(root.findall('Photogroup/SensorSize')[0].text)

    focal_x = Width*FocalLength /SensorSize
    focal_y = Height*FocalLength /SensorSize



    aspect_ratio = float(root.findall('Photogroup/AspectRatio')[0].text)
    camera_matrix = np.array([[focal_x, 0, camera[1]],
                              [0, focal_y, camera[2]],
                              [0, 0, 1]])
    distortion = np.array([float(root.findall('Photogroup/Distortion/K1')[0].text),
                           float(root.findall('Photogroup/Distortion/K2')[0].text),
                           float(root.findall('Photogroup/Distortion/P1')[0].text),
                           float(root.findall('Photogroup/Distortion/P2')[0].text),
                           float(root.findall('Photogroup/Distortion/K3')[0].text)])  # k1 k2 p1 p2 k3



    with (output_path / 'mappings.txt').open('w') as f:
        for i, rgb_name in enumerate(tqdm(images_name)):
            # if i % int(camera_positions.shape[0] / hparams.num_val) == 0:
            #     split_dir = output_path / 'val'
            # else:
            split_dir = output_path / 'train'
            (Path(split_dir)).mkdir(exist_ok=True)
            (Path(split_dir)/'metadata').mkdir(exist_ok=True)
            (Path(split_dir)/'rgbs').mkdir(exist_ok=True)
            if True:
                rgb_ori = str(images_path) + '/' + rgb_name
                rgb_ori = rgb_ori[:-4]+'.png'
                distorted = cv2.imread(rgb_ori)
                # undistorted = cv2.undistort(distorted, camera_matrix, distortion)#这里因为distortion为[0,0,0,0,0],所以等于没有做任何操作
                # cv2.imwrite(str(split_dir / 'rgbs' / '{0:06d}.jpg'.format(i)), undistorted)
                cv2.imwrite(str(split_dir / 'rgbs' / '{0:06d}.jpg'.format(i)), distorted)


            metadata_name = '{0:06d}.pt'.format(i)
            torch.save({
                'H': distorted.shape[0],
                'W': distorted.shape[1],
                # 'c2w': torch.cat([camera_in_drb[:, 1:2], -camera_in_drb[:, :1], camera_in_drb[:, 2:4]], -1),
                'c2w': torch.FloatTensor(c2w[i]),
                'intrinsics': torch.FloatTensor(
                    [camera_matrix[0][0], camera_matrix[1][1], camera_matrix[0][2], camera_matrix[1][2]]),
                'distortion': torch.FloatTensor([0,0,0,0,0]),
                # 'label': label
            }, split_dir / 'metadata' / metadata_name)

            f.write('{},{}\n'.format(rgb_name, metadata_name))

    print('Finish')


if __name__ == '__main__':
    main(_get_opts())
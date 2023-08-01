
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
    parser.add_argument('--output_path', default='dji/project_script/ply/output',type=str, required=False)
    parser.add_argument('--resume', default=True, action='store_false')  # debug
    # parser.add_argument('--resume', default=False, action='store_true')  # run
    parser.add_argument('--num_val', type=int, default=20, help='Number of images to hold out in validation set')
    parser.add_argument('--down_scale', type=int, default=1, help='')
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
    original_image_name_sorted = [x["original_image_name"] for x in sorted_process_data]

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

    
    # 质量报告里面的相机参数及畸变
    distortion_coeffs1 = np.array([float(-0.009831534),
                           float(0.013398247),
                           float(-0.002080232),
                           float(-0.001410897),
                           float(-0.010848353)])  # k1 k2 p1 p2 k3
    camera_matrix1 = np.array([[3695.607, 0, 2713.97],
                            [0, 3695.607, 1811.31],
                            [0, 0, 1]])
    #######-------------以上是dji/process_dji_v8_color.py中的内容，对pose和img进行了排序和转换-------------#########

    
    (output_path).mkdir(parents=True,exist_ok=True)

    # 读取点云数据和颜色信息
    points = np.load('dji/project_script/ply2_Mmerge/points.npy')
    colors = np.load('dji/project_script/ply2_Mmerge/colors.npy')

    # scale 变化
    coordinate_info = torch.load('/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/coordinates.pt')
    origin_drb = coordinate_info['origin_drb'].numpy()
    pose_scale_factor = coordinate_info['pose_scale_factor']

    for i, rgb_name in enumerate(tqdm(images_name_sorted)):
        if i %50 !=0:
        # if i != 350:
            continue
        if i % int(camera_positions.shape[0] / hparams.num_val) == 0:
            split_dir = 'val'
        else:
            split_dir = 'train'
        
        # img1 = cv2.imread(str(original_images_path) + '/' + original_image_name_sorted[i]) 
        # img_change = cv2.undistort(img1, camera_matrix1, distortion_coeffs1, None, camera_matrix)
        # cv2.imwrite(str(output_path)+'/'+'{0:06d}_1_rgbs.jpg'.format(i), img_change)


        metadata = torch.load('/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/' + split_dir + '/metadata/{0:06d}.pt'.format(i), map_location='cpu')
        

        #######################################
        ZYQ = torch.DoubleTensor([[0, 0, -1],
                                [0, 1, 0],
                                [1, 0, 0]])
        ZYQ_1 = torch.DoubleTensor([[1, 0, 0],
                                [0, math.cos(rad(135)), math.sin(rad(135))],
                                [0, -math.sin(rad(135)), math.cos(rad(135))]])      


        c2w_R = euler2rotation(camera_rotations[i])
        c2w_T = camera_positions[i]

        P = [[1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]
        
        ######### 这里的代码是debug用的##############
        # aaa = np.concatenate((c2w_R, c2w_T[:, np.newaxis]), axis=1)
        # fff = np.concatenate((aaa, [[0,0,0,1]]), axis=0)
        # bbb = (fff @ P)[:3, :]
        # ccc = ZYQ.numpy() @ bbb
        # ddd = ZYQ_1.numpy() @ ccc


        # temp = np.concatenate((c2w_R, c2w_T[:, np.newaxis]), axis=1)
        # temp = np.concatenate((temp[:,0:1], -temp[:,1:2], -temp[:,2:3], temp[:,3:]), axis=1)
        # temp = torch.hstack((ZYQ @ temp[:3, :3], ZYQ @ temp[:3, 3:]))
        # temp = torch.hstack((ZYQ_1 @ temp[:3, :3], ZYQ_1 @ temp[:3, 3:]))
        # temp = temp.numpy()
        # c2w = np.array(temp)
        ##########################




        points_nerf = points
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

        # 绘制投影点到图像上（根据颜色信息进行着色）
        count = 0
        mask_x1 = projected_points[0, :]>=0
        mask_x2 = projected_points[0, :]<=image_width
        mask_y1 = projected_points[1, :]>=0
        mask_y2 = projected_points[1, :]<=image_height

        mask = mask_x1 * mask_x2 * mask_y1 * mask_y2

        projected_points = projected_points[:, mask]
        pt_3d_trans_z = pt_3d_trans[2, mask]

        # 表示的是三维点到相机图像平面的距离, 不考虑相机的朝向
        depth_z = pt_3d_trans_z
        # 考虑相机朝向， 则用 三维投影点的（x,y）-> 相机中心的距离
        # depth_z = 

        for j in tqdm(range(projected_points.shape[1])):
            x, y = projected_points[:,j]
            x, y = int(x), int(y)
            if depth_z[j] < depth_map[y, x]:
                step = 2
                x_range = range(max(0, x-step), min(image_width, x+step))
                y_range = range(max(0, y-step), min(image_height, y+step))
                for x_i in x_range:
                    for y_i in y_range:
                        depth_map[y_i, x_i] = depth_z[j]
            count += 1
        # np.save(str(output_path)+'/'+'{0:06d}.npy'.format(i), depth_map)


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
        Image.fromarray(img_grid1).save(str(output_path)+'/'+'{0:06d}_3_depth.jpg'.format(i))


        img_list =[]
        depth_vis = np.load('/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/' + split_dir + '/depths/{0:06d}.npy'.format(i))
        
        min_depth = depth_vis.min()
        max_depth = depth_vis.max()

        depth_vis = (depth_vis - min_depth) / max(max_depth - min_depth, 1e-8)  # normalize to 0~1
        depth_vis = torch.from_numpy(depth_vis).clamp_(0, 1)
        depth_vis = ((1 - depth_vis) * 255).byte().numpy()  # inverse heatmap
        depth_vis = cv2.cvtColor(cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        
        # depth_vis = torch.from_numpy(visualize_scalars(
        #             torch.log(torch.from_numpy(depth_0) + 1e-8).cpu()))
        

        img_list.append(torch.from_numpy(depth_vis))
        img_list = torch.stack(img_list).permute(0,3,1,2)
        img = make_grid(img_list, nrow=3)
        img_grid = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        Image.fromarray(img_grid).save(str(output_path)+'/'+'{0:06d}_4_depth_nerf.jpg'.format(i))
        
    a = 1 

if __name__ == '__main__':
    main(_get_opts())
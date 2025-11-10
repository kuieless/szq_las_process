#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
这是一个合并后的脚本，用于：
1. (run_data_preparation): 将大疆智图(Terra)的XML和原始图像转换为NeRF格式数据。
2. (run_mesh_rendering):   加载一个.obj网格，将其渲染为NeRF相机的深度图。
"""

import argparse
import os
import sys
import torch
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import exifread
import math
from PIL import Image
from PIL.ExifTags import TAGS
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
import shutil
import trimesh
import random
from pathlib import Path
from bs4 import BeautifulSoup
import pytorch3d

# --- Pytorch3D 核心组件导入 ---
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    AmbientLights,
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRendererWithFragments, 
    MeshRasterizer, 
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    TexturesAtlas,
)

# --- 调试工具 ---
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(call_pdb=True)
from IPython.core.debugger import set_trace # debug

# ===================================================================
# --- 1. GPU 设备设置 ---
# ===================================================================

if torch.cuda.is_available():
    # 注意：这里硬编码了cuda:5。你可以根据需要修改，或者设为 cuda:0
    device = torch.device("cuda:5") 
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# ===================================================================
# --- 2. 辅助函数 (来自两个脚本) ---
# ===================================================================

### --- 来自 1_prepare_data.py ---

def rad(x):
    return math.radians(x)

def euler2rotation(theta, ):
    """ 
    将 Omega, Phi, Kappa 欧拉角转换为 3x3 旋转矩阵 (R_c_w)
    """
    theta = [rad(i) for i in theta]
    omega, phi, kappa = theta[0], theta[1], theta[2]

    R_omega = np.array([[1, 0, 0],
                        [0, math.cos(omega), -math.sin(omega)],
                        [0, math.sin(omega), math.cos(omega)]])

    R_phi = np.array([[math.cos(phi), 0, math.sin(phi)],
                        [0, 1, 0],
                        [-math.sin(phi), 0, math.cos(phi)]])

    R_kappa = np.array([[math.cos(kappa), -math.sin(kappa), 0],
                        [math.sin(kappa), math.cos(kappa), 0],
                        [0, 0, 1]])
    
    # 旋转顺序为 R_omega -> R_phi -> R_kappa
    R3d = R_omega @ R_phi @ R_kappa
    return R3d

### --- 来自 2_render_mesh_depth.py ---

def torch_depth_to_colormap_np(depth):
    """ (可视化用) Pytorch 深度图转彩色 """
    depth_normalized = torch.zeros(depth.shape, device=device)
    valid_mask = depth > -0.9 # valid
    if valid_mask.sum() > 0:
        d_valid = depth[valid_mask]
        depth_normalized[valid_mask] = (d_valid - d_valid.min()) / (d_valid.max() - d_valid.min())
        depth_np = (depth_normalized.cpu().numpy()[0] * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
        depth_normalized = depth_normalized.cpu().numpy()[0] * 255
    else:
        print('!!!! No depth projected !!!')
        depth_color = depth_normalized = np.zeros(depth.shape, dtype=np.uint8)
    return depth_color, depth_normalized

def np_depth_to_colormap(depth, normalize=True, thres=-1, min_depth_percentile=5, max_depth_percentile=95):
    """ (可视化用) Numpy 深度图转彩色 """
    depth_normalized = np.zeros(depth.shape)
    valid_mask = depth > -0.9 # depth for invalid pixel is -1
    if valid_mask.sum() > 0:
        d_valid = depth[valid_mask]
        if normalize:
            depth_normalized[valid_mask] = (d_valid - d_valid.min()) / (d_valid.max() - d_valid.min())
        elif thres > 0:
            depth_normalized[valid_mask] = d_valid.clip(0, thres) / thres
        elif min_depth_percentile>0:
            depth_normalized[valid_mask] = d_valid
            min_depth, max_depth = np.percentile(
                depth_normalized, [min_depth_percentile, max_depth_percentile])
            depth_normalized[depth_normalized < min_depth] = min_depth
            depth_normalized[depth_normalized > max_depth] = max_depth
            depth_normalized = depth_normalized / max_depth
        else:
            depth_normalized[valid_mask] = d_valid
        depth_np = (depth_normalized * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
        depth_normalized = depth_normalized
    else:
        print('!!!! No depth projected !!!')
        depth_color = depth_normalized = np.zeros((*depth.shape,3), dtype=np.uint8)
    return depth_color, depth_normalized

def compute_vert_v2(flat_vertices):
    """ (Mesh坐标系转换用) PCA找法线 """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(flat_vertices)
    normal_vector = pca.components_[2] # 第三个主成分即为法线向量
    return normal_vector

def matrix_rotateB2A(A, B):
    """ (Mesh坐标系转换用) 计算从A到B的旋转矩阵 """
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    rot_axis = np.cross(A, B)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    rot_angle = np.arccos(np.dot(A, B))
    a = np.cos(rot_angle / 2.0)
    b, c, d = -rot_axis * np.sin(rot_angle / 2.0) # flip the rot_axis
    rotation_matrix = np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]
    ])
    return rotation_matrix

def compute_MRM(flat_vertices):
    """ (Mesh坐标系转换用) """
    X_axis = np.array([1, 0, 0])
    target_axis = compute_vert_v2(flat_vertices)
    MRM = matrix_rotateB2A(X_axis, target_axis)
    return MRM

def load_mesh(args, obj_filename, coordinates, mesh_metadata_xml_path, save_mesh=False):
    """
    加载 .obj 文件, 并将其转换到NeRF坐标系
    """
    verts, faces, aux = load_obj(obj_filename, device=device,
                                 load_textures=True,
                                 create_texture_atlas=True,
                                 texture_atlas_size=8,
                                 texture_wrap=None)
    print('Loaded obj')
    atlas = aux.texture_atlas
    print('Verts before transform', verts.max(0)[0], verts.min(0)[0], verts.mean(0))

    if coordinates and mesh_metadata_xml_path:
        # 从 ModelMetadata.xml 加载模型的平移(SRSOrigin)
        root = ET.parse(mesh_metadata_xml_path).getroot()
        translation_str = root.find('SRSOrigin').text
        translation = np.array(translation_str.split(','), dtype=float)
        
        # 从 coordinates.pt 加载 NeRF 坐标系信息
        nerf_coords = {'coordinates': coordinates, 'translation': translation}
        verts = convert_verts_to_nerf_space(verts, nerf_coords)
        print('Verts converted to NeRF', verts.max(0)[0], verts.min(0)[0], verts.mean(0))
    else:
        print("Warning! No coordinates or mesh_metadata_xml. Normalizing mesh.")
        verts -= verts.mean(0, keepdim=True)
        verts /= verts.max()
        print('Verts normalized', verts.max(0)[0], verts.min(0)[0], verts.mean(0))

    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
        textures=TexturesAtlas(atlas=[atlas]),
    )     
    print('Created Pytorch3D Mesh object')
    
    if save_mesh:
        IO = pytorch3d.io.IO()
        save_path = os.path.join(args.save_dir, 'norm.obj')
        IO.save_mesh(mesh, save_path)
        print(f"Saved transformed mesh to {save_path}")
    return mesh

def load_nerf_metadata_for_render(args, metadata_dir, filename, verbose=True):
    """
    为渲染步骤加载单个图像的元数据
    """
    # 加载由 run_data_preparation 生成的 .pt 文件
    metadata_path = os.path.join(metadata_dir, args.split, 'metadata', f'{filename}.pt')
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        return None
    
    metadata = torch.load(metadata_path)
    
    # 加载对应的RGB图像
    rgb_path = os.path.join(metadata_dir, args.split, 'rgbs', f'{filename}.jpg')
    if not os.path.exists(rgb_path):
        print(f"Warning: RGB file not found: {rgb_path}")
        rgb = np.zeros((metadata['H'], metadata['W'], 3), dtype=np.uint8)
    else:
        rgb = cv2.imread(rgb_path)

    if verbose:
        print(f'Loaded metadata for {filename}:')
        print('  RGB shape:', rgb.shape)
        print('  c2w shape:', metadata['c2w'].shape)
        print('  Intrinsics:', metadata['intrinsics'])

    # 返回一个统一格式的字典
    return {'rgb': rgb, 'metadata': metadata}

def convert_verts_to_nerf_space(verts, nerf_coords):
    """
    执行从 .obj 坐标系 到 NeRF 坐标系的转换
    """
    origin_drb = nerf_coords['coordinates']['origin_drb'].to(device).to(torch.float64)
    pose_scale_factor = torch.tensor(nerf_coords['coordinates']['pose_scale_factor'].item()).to(torch.float64)
    translation = torch.from_numpy(nerf_coords['translation']).float().to(device)

    # 两个固定的旋转矩阵，用于对齐坐标系
    T1 = torch.FloatTensor([[0, 0, -1],
                            [0, 1, 0],
                            [1, 0, 0]]).to(device).to(torch.float64)
    T2 = torch.FloatTensor([[1, 0, 0],
                            [0, math.cos(rad(135)), math.sin(rad(135))],
                            [0, -math.sin(rad(135)), math.cos(rad(135))]]).to(device).to(torch.float64)

    # 步骤 1: 应用 ModelMetadata.xml 中的 SRSOrigin 平移
    verts += translation
    verts = verts.to(torch.float64)

    # 步骤 2: 应用固定的坐标系旋转 T1
    verts_nerf = T1 @ verts.T
    
    # 步骤 3: 应用固定的坐标系旋转 T2
    verts_nerf = (T2 @ verts_nerf).T

    # 步骤 4: 应用 NeRF 归一化 (平移到原点 + 缩放)
    verts_nerf = (verts_nerf - origin_drb) / pose_scale_factor
    print(f"Mesh transform: origin={origin_drb}, scale={pose_scale_factor}")
    return verts_nerf.to(torch.float32)

def setup_renderer(args, nerf_metadata, verbose=True):
    """
    根据 NeRF 元数据设置 Pytorch3D 的相机和渲染器
    """
    c2w = nerf_metadata['metadata']['c2w']
    R_nerf, T_nerf = c2w[:3, :3], c2w[:3, 3:]

    if verbose:
        print('Original NeRF c2w\n', torch.cat([R_nerf, T_nerf], 1))
    
    # 将 NeRF c2w 转换为 Pytorch3D W2C (World-to-Camera)
    # 1. NeRF (OpenCV/COLMAP) -> Pytorch3D 坐标系转换
    # (x, y, z) -> (x, -y, -z) 或其他，这里脚本用的是：
    # (x, y, z) -> (-x, y, -z)
    R_p3d = torch.stack([-R_nerf[:, 0], R_nerf[:, 1], -R_nerf[:, 2]], 1)
    T_p3d = T_nerf
    
    # 2. 构建 Pytorch3D 的 C2W 矩阵
    c2w_p3d = torch.cat([R_p3d, T_p3d], 1)
    if verbose:
        print('Pytorch3D-style c2w\n', c2w_p3d)

    # 3. 计算 W2C (World-to-Camera) 矩阵，即 C2W 的逆
    w2c_p3d = torch.linalg.inv(torch.cat((c2w_p3d, torch.Tensor([[0,0,0,1]])), 0))
    
    # 4. 提取 Pytorch3D 相机所需的 R 和 T
    R, T = w2c_p3d[:3, :3].permute(1, 0), w2c_p3d[:3, 3]
    if verbose:
        print('Final Pytorch3D w2c\n', w2c_p3d)

    R = R[None].to(device) # 增加 Batch 维度
    T = T[None].to(device) # 增加 Batch 维度

    # --- 设置相机内参 ---
    H, W = nerf_metadata['metadata']['H'], nerf_metadata['metadata']['W']
    H = int(H / args.down / args.simplify)
    W = int(W / args.down / args.simplify)

    # NeRF 的内参 [fx, fy, cx, cy]
    intrinsics = nerf_metadata['metadata']['intrinsics'] / args.down / args.simplify

    image_size_hw = ((H, W),)
    fcl_screen = ((intrinsics[0], intrinsics[1]),)
    prp_screen = ((intrinsics[2], intrinsics[3]), )
    
    # 创建相机对象
    cameras = PerspectiveCameras(focal_length=fcl_screen, principal_point=prp_screen, 
                                 in_ndc=False, image_size=image_size_hw, 
                                 R=R, T=T, device=device)
    
    image_size = (H, W)

    # --- 设置渲染器 ---
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        perspective_correct=True,
        bin_size=0
    )
    lights = AmbientLights(device=device)
    rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
    renderer = MeshRendererWithFragments(
        rasterizer = rasterizer,
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    
    render_setup = {'renderer': renderer}
    return render_setup

# (其他辅助函数，如 load_mesh_debug, convert_verts_to_nerf_space_debug 被省略)


# ===================================================================
# --- 3. 合并后的参数解析器 ---
# ===================================================================

def _get_opts():
    parser = argparse.ArgumentParser()

    # --- 步骤 1 (Data Prep) 的参数 ---
    parser.add_argument('--output_path', type=str, required=True, 
                        help='处理后 NeRF 数据的输出路径 (例如 ./nerf_data)')
    parser.add_argument('--original_images_path', type=str, required=True, 
                        help='包含所有原始JPG图像的文件夹路径')
    parser.add_argument('--infos_path', type=str, required=True, 
                        help='大疆智图的 BlocksExchange.xml 文件路径 (包含相机位姿)')
    parser.add_argument('--original_images_list_json_path', type=str, required=True, 
                        help='包含图像ID和路径映射的 .json 文件')
    parser.add_argument('--num_val', type=int, default=10, 
                        help='每 N 张图像中选1张作为验证集')

    # --- 步骤 1 的手动相机参数 (!! 关键 !!) ---
    parser.add_argument('--fx', type=float, required=True, help='优化的相机内参 fx')
    parser.add_argument('--fy', type=float, required=True, help='优化的相机内参 fy (通常等于fx)')
    parser.add_argument('--cx', type=float, required=True, help='优化的相机内参 cx')
    parser.add_argument('--cy', type=float, required=True, help='优化的相机内参 cy')
    parser.add_argument('--k1', type=float, default=0.0, help='优化的畸变参数 k1')
    parser.add_argument('--k2', type=float, default=0.0, help='优化的畸变参数 k2')
    parser.add_argument('--k3', type=float, default=0.0, help='优化的畸变参数 k3')
    parser.add_argument('--p1', type=float, default=0.0, help='优化的畸变参数 p1')
    parser.add_argument('--p2', type=float, default=0.0, help='优化的畸变参数 p2')

    # --- 步骤 2 (Mesh Render) 的参数 ---
    parser.add_argument('--obj_filename', type=str, required=True, 
                        help='要渲染的 .obj 模型文件路径')
    parser.add_argument('--mesh_metadata_xml', type=str, required=True, 
                        help='模型的 ModelMetadata.xml 文件路径 (包含SRSOrigin)')
    parser.add_argument('--down', type=float, default=1.0, 
                        help='渲染时的图像下采样倍数')
    parser.add_argument('--simplify', type=int, default=1, 
                        help='(另一个下采样因子, 保持1即可)')
    parser.add_argument('--camera_type', default='perspective', 
                        help='相机类型 (保持 perspective 不变)')
    parser.add_argument('--save_mesh', action='store_true', 
                        help='是否保存转换到NeRF坐标系后的 .obj 模型')
    parser.add_argument('--visualize', action='store_true', 
                        help='是否保存渲染结果的可视化对比图')
    parser.add_argument('--debug', action='store_true', 
                        help='是否开启调试模式')

    return parser.parse_args()


# ===================================================================
# --- 4. 步骤 1: 数据预处理 (来自 1_prepare_data.py) ---
# ===================================================================

def run_data_preparation(args):
    print("--- [步骤 1] 开始：将 DJI XML 转换为 NeRF 格式 ---")
    
    # 路径初始化
    output_path = Path(args.output_path)
    original_images_path = Path(args.original_images_path)
    original_images_list_json_path = Path(args.original_images_list_json_path)

    # 1. 从 BlocksExchange.xml 加载位姿和图像名称
    print(f"正在读取相机位姿: {args.infos_path}")
    root = ET.parse(args.infos_path).getroot()
    xml_pose = np.array([[ float(pose.find('Center/x').text),
                           float(pose.find('Center/y').text),
                           float(pose.find('Center/z').text),
                           float(pose.find('Rotation/Omega').text),
                           float(pose.find('Rotation/Phi').text),
                           float(pose.find('Rotation/Kappa').text)] 
                           for pose in root.findall('Block/Photogroups/Photogroup/Photo/Pose')])
    images_name = [Images_path.text.split("\\")[-1] 
                   for Images_path in root.findall('Block/Photogroups/Photogroup/Photo/ImagePath')]
    print(f"从 XML 中加载了 {len(xml_pose)} 个位姿。")

    # 2. 加载 .json 列表进行匹配和排序
    print(f"正在读取 JSON 图像列表: {original_images_list_json_path}")
    with open(original_images_list_json_path, "r") as file:
        json_data = json.load(file)

    sorted_json_data = sorted(json_data, key=lambda x: x["origin_path"])
    sorted_process_data = []
    
    for json_line in tqdm(sorted_json_data, desc="匹配 XML 和 JSON"):
        sorted_process_line = {}
        id = json_line['id']
        
        # 匹配 XML 和 JSON 中的图像
        if (id+'.jpg') in images_name:
            index = images_name.index(id+'.jpg')
        else:
            continue # 跳过在 XML 中找不到的图像

        sorted_process_line['images_name'] = images_name[index]
        
        path_segments = json_line['origin_path'].split('/')
        last_two_path = '/'.join(path_segments[-2:]) # e.g., "100_0001/DJI_0001.jpg"
        sorted_process_line['original_image_name'] = last_two_path
        sorted_process_line['pose'] = xml_pose[index, :]
        
        # 提取 EXIF (可选, 但脚本里有)
        image_path = os.path.join(args.original_images_path, last_two_path)
        if os.path.exists(image_path):
            with open(image_path, 'rb') as image_file:
                tags = exifread.process_file(image_file, details=False)
            tags['original_image_path'] = last_two_path
            if 'JPEGThumbnail' in tags: del tags['JPEGThumbnail']
            sorted_process_line['meta_tags'] = tags
        else:
            print(f"Warning: 找不到原始图像 {image_path}")
            sorted_process_line['meta_tags'] = {}

        sorted_process_data.append(sorted_process_line)
    
    print(f"成功匹配 {len(sorted_process_data)} 张图像。")

    # 3. 将位姿转换为 NeRF (c2w) 格式
    xml_pose_sorted = np.array([x["pose"] for x in sorted_process_data])
    images_name_sorted = [x["images_name"] for x in sorted_process_data]
    original_image_name_sorted = [x["original_image_name"] for x in sorted_process_data]

    camera_positions = xml_pose_sorted[:, 0:3]
    camera_rotations = xml_pose_sorted[:, 3:6]

    # (欧拉角 -> 旋转矩阵)
    c2w_R = []
    for i in range(len(camera_rotations)):
        R_temp = euler2rotation(camera_rotations[i])
        c2w_R.append(R_temp)

    # 固定的坐标系转换矩阵 (与 convert_verts_to_nerf_space 中的 T1, T2 对应)
    ZYQ = torch.DoubleTensor([[0, 0, -1],
                              [0, 1, 0],
                              [1, 0, 0]])
    ZYQ_1 = torch.DoubleTensor([[1, 0, 0],
                                [0, math.cos(rad(135)), math.sin(rad(135))],
                                [0, -math.sin(rad(135)), math.cos(rad(135))]])

    c2w = []
    for i in range(len(c2w_R)):
        # 1. 合并 R 和 T (Center)
        temp = np.concatenate((c2w_R[i], camera_positions[i:i + 1].T), axis=1)
        
        # 2. 转换坐标系 (OpenCV -> OpenGL)
        # (x, y, z) -> (x, -y, -z)
        temp = np.concatenate((temp[:,0:1], -temp[:,1:2], -temp[:,2:3], temp[:,3:]), axis=1)
        
        # 3. 应用固定旋转 ZYQ (T1)
        temp = torch.hstack((ZYQ @ temp[:3, :3], ZYQ @ temp[:3, 3:]))
        
        # 4. 应用固定旋转 ZYQ_1 (T2)
        temp = torch.hstack((ZYQ_1 @ temp[:3, :3], ZYQ_1 @ temp[:3, 3:]))
        c2w.append(temp.numpy())
    
    c2w = np.array(c2w)

    # 4. 归一化场景
    min_position = np.min(c2w[:,:, 3], axis=0)
    max_position = np.max(c2w[:,:, 3], axis=0)
    print(f'NeRF 坐标系范围 (变换后): Min={min_position}, Max={max_position}')
    
    origin = (max_position + min_position) * 0.5
    dist = (torch.tensor(c2w[:,:, 3]) - torch.tensor(origin)).norm(dim=-1)
    diagonal = dist.max()
    scale = diagonal.numpy()
    
    c2w[:,:, 3] = (c2w[:,:, 3] - origin) / scale # 归一化相机位置
    print(f"NeRF 场景: origin={origin}, scale={scale}")
    assert np.logical_and(c2w[:,:, 3] >= -1, c2w[:,:, 3] <= 1).all()

    # 5. 创建输出目录
    if not os.path.exists(args.output_path):
        output_path.mkdir(parents=True)
    (output_path / 'train' / 'metadata').mkdir(parents=True, exist_ok=True)
    (output_path / 'train' / 'rgbs').mkdir(parents=True, exist_ok=True)
    (output_path / 'val' / 'metadata').mkdir(parents=True, exist_ok=True)
    (output_path / 'val' / 'rgbs').mkdir(parents=True, exist_ok=True)
    # (可选: image_metadata 文件夹)
    (output_path / 'train' / 'image_metadata').mkdir(parents=True, exist_ok=True)
    (output_path / 'val' / 'image_metadata').mkdir(parents=True, exist_ok=True)

    # 6. 保存 coordinates.pt
    coordinates = {
        'origin_drb': torch.Tensor(origin),
        'pose_scale_factor': scale
    }
    torch.save(coordinates, output_path / 'coordinates.pt')

    # 7. 从 XML 加载 *原始* 内参 (用于 undistort 的目标)
    camera = np.array([float(root.findall('Block/Photogroups/Photogroup/FocalLengthPixels')[0].text),
                       float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/x')[0].text),
                       float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/y')[0].text)])
    aspect_ratio = float(root.findall('Block/Photogroups/Photogroup/AspectRatio')[0].text)
    
    # 这是 NeRF 最终使用的内参矩阵 (来自 XML)
    camera_matrix = np.array([[camera[0], 0, camera[1]],
                              [0, camera[0]*aspect_ratio, camera[2]],
                              [0, 0, 1]])
    
    # 这是 XML 中的畸变 (通常为0，因为 Terra 默认使用已去畸变的图)
    distortion = np.array([float(root.findall('Block/Photogroups/Photogroup/Distortion/K1')[0].text),
                           float(root.findall('Block/Photogroups/Photogroup/Distortion/K2')[0].text),
                           float(root.findall('Block/Photogroups/Photogroup/Distortion/P1')[0].text),
                           float(root.findall('Block/Photogroups/Photogroup/Distortion/P2')[0].text),
                           float(root.findall('Block/Photogroups/Photogroup/Distortion/K3')[0].text)])

    # 8. 这是你 *手动输入* 的优化后的内参和畸变
    distortion_manual = np.array([float(args.k1),
                                  float(args.k2),
                                  float(args.p1),
                                  float(args.p2),
                                  float(args.k3)])
    camera_matrix_manual = np.array([[args.fx, 0, args.cx],
                                     [0, args.fy, args.cy], # 注意: 这里用了 args.fx (原脚本错误?)，我改成了 args.fy
                                     [0, 0, 1]])
    
    print("--- 应用以下相机参数 ---")
    print("手动内参 (用于读取原图):")
    print(camera_matrix_manual)
    print("手动畸变 (用于读取原图):")
    print(distortion_manual)
    print("目标内参 (来自XML, NeRF使用):")
    print(camera_matrix)
    print("-------------------------")

    # 9. 循环处理每张图：去畸变、保存 .pt 和 .jpg
    with (output_path / 'mappings.txt').open('w') as f:
        for i, rgb_name in enumerate(tqdm(images_name_sorted, desc="处理和保存图像")):
            
            # 分配 train/val
            if i % args.num_val == 0:
                split_dir = output_path / 'val'
            else:
                split_dir = output_path / 'train'
            
            # 读取原始图像
            img_path = os.path.join(args.original_images_path, original_image_name_sorted[i])
            if not os.path.exists(img_path):
                print(f"Skipping {img_path}, file not found.")
                continue
            
            img_original = cv2.imread(img_path)
            
            # **核心步骤：图像去畸变**
            # 使用手动的内参/畸变(camera_matrix_manual, distortion_manual)读取原图,
            # 并将其转换为目标内参(camera_matrix)的无畸变图像。
            img_undistorted = cv2.undistort(img_original, camera_matrix_manual, distortion_manual, None, camera_matrix)
            
            # 保存处理后的图像
            img_save_name = '{0:06d}.jpg'.format(i)
            cv2.imwrite(str(split_dir / 'rgbs' / img_save_name), img_undistorted)

            # 保存 NeRF metadata (.pt)
            metadata_name = '{0:06d}.pt'.format(i)
            torch.save({
                'H': img_original.shape[0],
                'W': img_original.shape[1],
                'c2w': torch.FloatTensor(c2w[i]),
                'intrinsics': torch.FloatTensor(
                    [camera_matrix[0][0], camera_matrix[1][1], camera_matrix[0][2], camera_matrix[1][2]]),
                'distortion': torch.FloatTensor(distortion), # 保存来自XML的畸变(通常为0)
            }, split_dir / 'metadata' / metadata_name)

            f.write(f'{rgb_name},{metadata_name}\n')

            # (可选) 保存原始 metadata
            torch.save(sorted_process_data[i], split_dir / 'image_metadata' / '{0:06d}.pt'.format(i))

    print(f"--- [步骤 1] 完成！NeRF 数据已保存到: {args.output_path} ---")


# ===================================================================
# --- 5. 步骤 2: 网格渲染 (来自 2_render_mesh_depth.py) ---
# ===================================================================

def run_mesh_rendering(args):
    print(f"--- [步骤 2] 开始：为 '{args.split}' 集渲染深度图 ---")
    
    save_dir = Path(args.save_dir)
    
    # 1. 创建输出文件夹
    if not os.path.exists(save_dir / 'cat_results_' / args.split) and args.visualize:
        (save_dir / f'cat_results_{args.split}').mkdir(parents=True, exist_ok=True)
    if not os.path.exists(save_dir / args.split / 'depth_mesh'):
        (save_dir / args.split / 'depth_mesh').mkdir(parents=True, exist_ok=True)

    # 2. 加载 NeRF 坐标系
    coord_path = save_dir / 'coordinates.pt'
    if not os.path.exists(coord_path):
        print(f"错误: 找不到 coordinates.pt, 无法对齐网格。请先运行步骤1。")
        return
    coordinates = torch.load(coord_path)

    # 3. 加载并转换 .obj 网格
    print("正在加载和转换网格...")
    mesh = load_mesh(args, 
                     args.obj_filename, 
                     coordinates, 
                     args.metadataXml_path, # 注意: 这里用的是 --mesh_metadata_xml
                     save_mesh=args.save_mesh)
    print("网格加载并对齐完毕。")

    # 4. 查找所有要渲染的图像
    rgb_glob = str(save_dir / args.split / 'rgbs' / '*.jpg')
    names = sorted(glob.glob(rgb_glob))
    if not names:
        print(f"错误: 在 {rgb_glob} 中找不到任何 .jpg 图像。")
        return
        
    print(f"开始为 {len(names)} 张图像（'{args.split}'集）渲染深度...")

    # 5. 循环渲染
    for name in tqdm(names[:], desc=f"渲染 {args.split} 集"):
        filename = os.path.basename(name)[:-4] # e.g., "000001"
        
        # A. 加载这张图的 NeRF 元数据
        nerf_metadata = load_nerf_metadata_for_render(args, args.nerf_metadata_dir, filename, verbose=False)
        if nerf_metadata is None:
            continue
            
        # B. 设置此相机的渲染器
        render_setup = setup_renderer(args, nerf_metadata, verbose=False)

        # C. 执行渲染
        renderer = render_setup['renderer']
        images, fragments = renderer(mesh)
        rgb_np = images[0, ..., :3].cpu().numpy()[:, :, ::-1] # 渲染的彩色图 (可视化用)

        # D. 提取深度图 (NeRF尺度)
        depth_nerf_scale = fragments.zbuf[0, :, :, 0].cpu().numpy()

        # --- !! 关键修改：转换到真实尺度 !! ---
        
        # 1. 从已加载的 'coordinates' 字典中获取缩放因子
        #    (coordinates 是在函数开头加载的: torch.load(coord_path))
        scale_factor = coordinates['pose_scale_factor'].item()

        # 2. 创建一个新的数组来存储真实尺度的深度
        #    我们用 1e6 (或你喜欢的无效值) 来填充
        depth_real_scale = np.full_like(depth_nerf_scale, 1e6, dtype=np.float32)

        # 3. 找到所有有效像素 (zbuf > 0 的地方)
        valid_mask = (depth_nerf_scale > 0)

        # 4. 仅对有效像素应用缩放，将其从 "NeRF尺度" 转换回 "真实尺度"
        depth_real_scale[valid_mask] = depth_nerf_scale[valid_mask] * scale_factor
        
        # --- !! 修改结束 !! ---


        # E. 保存真实尺度的深度 .npy
        if args.simplify == 1:
            save_path = save_dir / args.split / 'depth_mesh' / f'{filename}.npy'
            
            # 保存真实尺度的 'depth_real_scale'
            # 注意：我们使用 float32 来保证精度，因为真实尺度（米）可能超过 float16 的范围
            np.save(save_path, depth_real_scale[:,:,None].astype(np.float32))

        # F. (可选) 可视化
        if args.visualize:
            # 注意：可视化时，我们仍然使用 'depth_nerf_scale' (NeRF尺度) 来上色
            # 因为 'np_depth_to_colormap' 期望的是一个可以被归一化的范围
            # 如果用真实尺度(如 50m 到 150m)，上色效果会很差
            
            depth_vis = depth_nerf_scale.copy()
            depth_vis[depth_vis <= 0] = -1 # 将所有无效像素(0 或 -1) 标记为 -1
            
            # normalize=True 会自动处理归一化，生成好看的可视化图
            depth_color, _ = np_depth_to_colormap(depth_vis, normalize=True)
            
            # 加载原始(已去畸变)的RGB图
            ori_image = nerf_metadata['rgb']
            ori_image = cv2.resize(ori_image, (rgb_np.shape[1], rgb_np.shape[0]))
            
            # (这里 lidar_depth_color 被设为和 depth_color 一样, 因为我们没有真值)
            lidar_depth_color = depth_color 
            image_diff_color = np.zeros_like(ori_image)
            depth_diff_color = np.zeros_like(ori_image)
            
            # 拼接: [ (渲染RGB, 渲染Depth), (真实RGB, 渲染Depth) ]
            results_1 = np.concatenate([rgb_np * 255, depth_color], 1)
            results_2 = np.concatenate([ori_image, lidar_depth_color], 1)
            results_3 = np.concatenate([image_diff_color, depth_diff_color], 1)
            cat_results = np.concatenate([results_1, results_2, results_3], 0)
            
            vis_save_path = save_dir / f'cat_results_{args.split}' / f'{filename}_cat_results.jpg'
            cv2.imwrite(str(vis_save_path), cat_results.astype(np.uint8))
            
    print(f"--- [步骤 2] '{args.split}' 集渲染完成！---")


# ===================================================================
# --- 6. 主程序入口 ---
# ===================================================================

if __name__ == '__main__':
    args = _get_opts()

    # --- 步骤 1: 准备 NeRF 数据 ---
    try:
        run_data_preparation(args)
    except Exception as e:
        print(f"错误：步骤 1 (数据准备) 失败: {e}")
        sys.exit(1)

    # --- 步骤 2: 渲染深度图 ---
    
    # 为步骤2准备参数映射
    args.nerf_metadata_dir = args.output_path
    args.save_dir = args.output_path
    args.metadataXml_path = args.mesh_metadata_xml # 关键映射
    
    # 渲染 'train' 集
    try:
        args.split = 'train'
        run_mesh_rendering(args)
    except Exception as e:
        print(f"错误：步骤 2 (渲染 'train' 集) 失败: {e}")

    # 渲染 'val' 集
    try:
        args.split = 'val'
        run_mesh_rendering(args)
    except Exception as e:
        print(f"错误：步骤 2 (渲染 'val' 集) 失败: {e}")

    print("================================")
    print("=== 所有步骤执行完毕 ===")
    print(f" NeRF 数据和深度图已保存到: {args.output_path}")
    print("================================")


    '''
python szq_GES_render_mesh_depth-terra.py \
    --output_path /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/output2 \
    --original_images_path /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/rgbsed \
    --infos_path /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/BlocksExchangeUndistortAT.xml \
    --original_images_list_json_path /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/rgbsed.json \
    --obj_filename /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/Mergedmesh.obj \
    --mesh_metadata_xml /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/metadata.xml \
    --num_val 10 \
    --visualize \
    \
    --fx 935.58 \
    --fy 935.58 \
    --cx 959.27 \
    --cy 540.00 \
    --k1 -0.00007137 \
    --k2  0.00031168 \
    --k3 -0.00023027 \
    --p1 0.00000496 \
    --p2 -0.00006902


    '''
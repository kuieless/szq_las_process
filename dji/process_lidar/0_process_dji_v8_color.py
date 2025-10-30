

# 脚本功能：
# 1. (新增) 在每个关键步骤后添加详细的打印语句，方便用户追踪数据流。
# 2. (新增) 在畸变校正步骤后，自动生成并保存一张“原始图像 vs 校正后图像”的对比图。
# 3. (新增) 在坐标变换后，打印位姿的范围和缩放因子，帮助理解场景尺度。
# 4. 保留并优化了XML和JSON文件之间的数据匹配逻辑。

from argparse import Namespace
from pathlib import Path
import numpy as np
import configargparse
import os
from tqdm import tqdm
import shutil
import torch
import random
import math
import sys
import cv2
import xml.etree.ElementTree as ET  # <-- Typo corrected here
import json
from PIL import Image
import exifread

# --- 项目路径设置 (与您的脚本保持一致) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)
from dji.opt import _get_opts


# --- 辅助函数 (保持不变) ---
def rad(x):
    return math.radians(x)


def euler2rotation(theta):
    theta = [rad(i) for i in theta]
    omega, phi, kappa = theta[0], theta[1], theta[2]
    R_omega = np.array([[1, 0, 0], [0, math.cos(omega), -math.sin(omega)], [0, math.sin(omega), math.cos(omega)]])
    R_phi = np.array([[math.cos(phi), 0, math.sin(phi)], [0, 1, 0], [-math.sin(phi), 0, math.cos(phi)]])
    R_kappa = np.array([[math.cos(kappa), -math.sin(kappa), 0], [math.sin(kappa), math.cos(kappa), 0], [0, 0, 1]])
    return R_omega @ R_phi @ R_kappa


def main(hparams):
    # ==============================================================================
    # --- 1. 初始化和路径设置 ---
    # ==============================================================================
    print("\n" + "=" * 20 + " 步骤 1: 初始化和路径设置 " + "=" * 20)
    output_path = Path(hparams.dataset_path)
    original_images_path = Path(hparams.original_images_path)

    # (新增) 创建一个专门存放调试输出的文件夹
    debug_output_path = output_path / 'debug_outputs'
    debug_output_path.mkdir(parents=True, exist_ok=True)
    print(f"脚本启动，所有调试文件将保存在: {debug_output_path}")

    # ==============================================================================
    # --- 2. 从 XML 和 JSON 加载并匹配数据 ---
    # ==============================================================================
    print("\n" + "=" * 20 + " 步骤 2: 加载并匹配元数据 " + "=" * 20)
    # 从 .xml 加载位姿和文件名
    root = ET.parse(hparams.infos_path).getroot()
    xml_pose = np.array([[float(pose.find('Center/x').text), float(pose.find('Center/y').text),
                          float(pose.find('Center/z').text), float(pose.find('Rotation/Omega').text),
                          float(pose.find('Rotation/Phi').text), float(pose.find('Rotation/Kappa').text)] for pose in
                         root.findall('Block/Photogroups/Photogroup/Photo/Pose')])
    images_name = [Images_path.text.split("\\")[-1] for Images_path in
                   root.findall('Block/Photogroups/Photogroup/Photo/ImagePath')]
    print(f"从 XML 文件 '{hparams.infos_path}' 中加载了 {len(images_name)} 条位姿记录。")

    # 从 .json 加载原始图像列表
    with open(hparams.original_images_list_json_path, "r") as file:
        json_data = json.load(file)
    print(f"从 JSON 文件 '{hparams.original_images_list_json_path}' 中加载了 {len(json_data)} 条图像记录。")

    # 按原始路径排序 JSON 数据 (这是一个好习惯)
    sorted_json_data = sorted(json_data, key=lambda x: x["origin_path"])

    # 执行匹配
    sorted_process_data = []
    for json_line in tqdm(sorted_json_data, desc="正在匹配 JSON 和 XML 数据"):
        id = json_line['id']
        target_image_name = id + '.jpg'
        if target_image_name in images_name:
            index = images_name.index(target_image_name)

            path_segments = json_line['origin_path'].split('/')
            last_two_path = '/'.join(path_segments[-2:])

            sorted_process_data.append({
                'images_name': images_name[index],
                'original_image_name': last_two_path,
                'pose': xml_pose[index, :]
            })

    print(f"匹配完成！在 JSON 和 XML 中共找到 {len(sorted_process_data)} 张共同的图像。")
    if len(sorted_process_data) == 0:
        print("❌ 严重错误: 没有匹配到任何数据，请检查 XML 和 JSON 文件是否来自同一次任务！")
        sys.exit(1)

    xml_pose_sorted = np.array([x["pose"] for x in sorted_process_data])
    images_name_sorted = [x["images_name"] for x in sorted_process_data]
    original_image_name_sorted = [x["original_image_name"] for x in sorted_process_data]

    # ==============================================================================
    # --- 3. 位姿坐标变换 ---
    # ==============================================================================
    print("\n" + "=" * 20 + " 步骤 3: 转换位姿坐标系 " + "=" * 20)
    pose_dji = xml_pose_sorted
    camera_positions = pose_dji[:, 0:3]
    camera_rotations = pose_dji[:, 3:6]

    c2w_R = [euler2rotation(rot) for rot in camera_rotations]

    ZYQ = torch.DoubleTensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    ZYQ_1 = torch.DoubleTensor(
        [[1, 0, 0], [0, math.cos(rad(135)), math.sin(rad(135))], [0, -math.sin(rad(135)), math.cos(rad(135))]])

    c2w = []
    for i in range(len(c2w_R)):
        temp = np.concatenate((c2w_R[i], camera_positions[i:i + 1].T), axis=1)
        temp = np.concatenate((temp[:, 0:1], -temp[:, 1:2], -temp[:, 2:3], temp[:, 3:]), axis=1)
        temp = torch.hstack((ZYQ @ temp[:3, :3], ZYQ @ temp[:3, 3:]))
        temp = torch.hstack((ZYQ_1 @ temp[:3, :3], ZYQ_1 @ temp[:3, 3:]))
        c2w.append(temp.numpy())
    c2w = np.array(c2w)

    min_position = np.min(c2w[:, :, 3], axis=0)
    max_position = np.max(c2w[:, :, 3], axis=0)
    origin = (max_position + min_position) * 0.5
    dist = torch.norm(torch.tensor(c2w[:, :, 3]) - torch.tensor(origin), dim=-1)
    scale = dist.max().item()
    c2w[:, :, 3] = (c2w[:, :, 3] - origin) / scale

    print("位姿变换和归一化完成。")
    print(f"  - 场景中心点 (origin): {np.round(origin, 2)}")
    print(f"  - 场景缩放因子 (scale): {scale:.4f}")
    print(f"  - 归一化后坐标范围: Z_min={np.min(c2w[:, 2, 3]):.2f}, Z_max={np.max(c2w[:, 2, 3]):.2f}")

    # ==============================================================================
    # --- 4. 创建目录结构和保存坐标信息 ---
    # ==============================================================================
    print("\n" + "=" * 20 + " 步骤 4: 创建输出目录并保存坐标信息 " + "=" * 20)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'train' / 'image_metadata').mkdir(parents=True, exist_ok=True)
    (output_path / 'train' / 'metadata').mkdir(parents=True, exist_ok=True)
    (output_path / 'train' / 'rgbs').mkdir(parents=True, exist_ok=True)
    (output_path / 'val' / 'image_metadata').mkdir(parents=True, exist_ok=True)
    (output_path / 'val' / 'metadata').mkdir(parents=True, exist_ok=True)
    (output_path / 'val' / 'rgbs').mkdir(parents=True, exist_ok=True)
    print("输出目录结构创建/确认完毕。")

    coordinates = {'origin_drb': torch.Tensor(origin), 'pose_scale_factor': scale}
    torch.save(coordinates, output_path / 'coordinates.pt')
    print(f"坐标信息已保存至: {output_path / 'coordinates.pt'}")

    # ==============================================================================
    # --- 5. (核心调试) 畸变校正与文件保存 ---
    # ==============================================================================
    print("\n" + "=" * 20 + " 步骤 5: 图像畸变校正与保存 " + "=" * 20)
    camera_xml = np.array([float(root.findall('Block/Photogroups/Photogroup/FocalLengthPixels')[0].text),
                           float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/x')[0].text),
                           float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/y')[0].text)])
    aspect_ratio_xml = float(root.findall('Block/Photogroups/Photogroup/AspectRatio')[0].text)
    camera_matrix_xml = np.array(
        [[camera_xml[0], 0, camera_xml[1]], [0, camera_xml[0] * aspect_ratio_xml, camera_xml[2]], [0, 0, 1]])

    distortion_optimized = np.array(
        [float(hparams.k1), float(hparams.k2), float(hparams.p1), float(hparams.p2), float(hparams.k3)])
    camera_matrix_optimized = np.array([[hparams.fx, 0, hparams.cx], [0, hparams.fx, hparams.cy], [0, 0, 1]])

    print("将使用以下优化后的相机参数进行畸变校正：")
    print(f"  - 新相机矩阵 (K_optimized):\n{np.round(camera_matrix_optimized, 2)}")
    print(f"  - 新畸变系数 (D_optimized): {np.round(distortion_optimized, 6)}")
    print(f"  - 目标相机矩阵 (newCameraMatrix) 将使用XML中的原始值:\n{np.round(camera_matrix_xml, 2)}")

    for i, data_item in enumerate(tqdm(sorted_process_data, desc="正在处理和保存每张图像")):
        if i % int(len(sorted_process_data) / hparams.num_val) == 0:
            split_dir = output_path / 'val'
        else:
            split_dir = output_path / 'train'

        original_image_path = original_images_path / data_item['original_image_name']
        img_original = cv2.imread(str(original_image_path))
        if img_original is None:
            print(f"警告：无法读取图像 {original_image_path}，跳过此文件。")
            continue

        img_undistorted = cv2.undistort(img_original, camera_matrix_optimized, distortion_optimized, None,
                                        camera_matrix_xml)

        output_rgb_path = split_dir / 'rgbs' / f'{i:06d}.jpg'
        cv2.imwrite(str(output_rgb_path), img_undistorted)

        if i == 0 or i == len(sorted_process_data) // 2 or i == len(sorted_process_data) - 1:
            print(f"\n正在为图像 {i} 生成畸变校正对比图...")
            h, w, _ = img_original.shape
            img_original_small = cv2.resize(img_original, (w // 4, h // 4))
            img_undistorted_small = cv2.resize(img_undistorted, (w // 4, h // 4))
            cv2.putText(img_original_small, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(img_undistorted_small, "Undistorted", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            comparison_image = np.hstack((img_original_small, img_undistorted_small))
            comparison_path = debug_output_path / f'undistort_comparison_{i:06d}.jpg'
            cv2.imwrite(str(comparison_path), comparison_image)
            print(f"对比图已保存至: {comparison_path}")

        # --- *** 错误修正处 *** ---
        # 提取、组合并保存所有元数据
        metadata_name = f'{i:06d}.pt'

        # 1. 提取原始EXIF数据
        with open(original_image_path, 'rb') as image_file:
            tags = exifread.process_file(image_file, details=False, stop_tag='JPEGThumbnail')
        if 'JPEGThumbnail' in tags: del tags['JPEGThumbnail']

        # 2. 保存NeRF训练所需的metadata
        torch.save({
            'H': img_original.shape[0], 'W': img_original.shape[1],
            'c2w': torch.FloatTensor(c2w[i]),
            'intrinsics': torch.FloatTensor(
                [camera_matrix_xml[0, 0], camera_matrix_xml[1, 1], camera_matrix_xml[0, 2], camera_matrix_xml[1, 2]]),
        }, split_dir / 'metadata' / metadata_name)

        # 3. (已修复) 保存后续脚本所需的 image_metadata
        #    它包含初始匹配信息和完整的EXIF标签
        full_metadata_to_save = data_item.copy()
        full_metadata_to_save['meta_tags'] = tags
        torch.save(full_metadata_to_save, split_dir / 'image_metadata' / metadata_name)
        # --- *** 修正结束 *** ---

    print("\n" + "=" * 25 + " 脚本执行完毕 " + "=" * 25)


if __name__ == '__main__':
    main(_get_opts())

#SBMU     #python 0_process_dji_v8_color.py --dataset_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output --original_images_path F:\download\SMBU2\smbu11161103\images\survey --infos_path F:\download\SMBU2\smbu11161103\AT\BlocksExchangeUndistortAT.xml --original_images_list_json_path F:\download\SMBU2\smbu11161103\images\survey\image_list.json --num_val 10 --fx 3691.094 --fy 3691.094 --cx 2755.620 --cy 1797.584 --k1 0.002214303 --k2 -0.009188852 --p1 -0.002608224 --p2 0.001401760 --k3 0.007309819


 #SZTTI   python 0_process_dji_v8_color.py --dataset_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output --original_images_path H:\BaiduNetdiskDownload\LiDAR_Raw\SZIIT\PSZIIT018\images\survey --infos_path H:\BaiduNetdiskDownload\LiDAR_Raw\SZIIT\PSZIIT018\AT\BlocksExchangeUndistortAT.xml --original_images_list_json_path H:\BaiduNetdiskDownload\LiDAR_Raw\SZIIT\PSZIIT018\images\survey\image_list.json --num_val 10 --fx 3692.240 --fy 3691.094 --cx 2755.094 --cy 1796.394 --k1 0.001630065 --k2 -0.007661543 --p1 -0.002702158 --p2 0.001345463 --k3 0.006244946

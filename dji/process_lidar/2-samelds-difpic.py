# 新脚本: 3_project_time_slice_to_all_images.py

import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2
from pathlib import Path
import json
import math
import torch
import os
import sys
import laspy
import re

# --- 项目路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)
from dji.opt import _get_opts


# --- 辅助函数 (与脚本2相同) ---
def rad(x): return math.radians(x)


def euler2rotation(theta):
    theta = [rad(i) for i in theta]
    omega, phi, kappa = theta[0], theta[1], theta[2]
    R_omega = np.array([[1, 0, 0], [0, math.cos(omega), -math.sin(omega)], [0, math.sin(omega), math.cos(omega)]])
    R_phi = np.array([[math.cos(phi), 0, math.sin(phi)], [0, 1, 0], [-math.sin(phi), 0, math.cos(phi)]])
    R_kappa = np.array([[math.cos(kappa), -math.sin(kappa), 0], [math.sin(kappa), math.cos(kappa), 0], [0, 0, 1]])
    return R_omega @ R_phi @ R_kappa


def main(hparams):
    # ==============================================================================
    # --- 用户配置与控制区域 ---
    # 1. 指定要提取的LiDAR点云的绝对GPS时间范围
    #    请注意: 这里的GPS时间是真实的时间戳，不是相对于文件开始的相对时间
    LIDAR_START_TIME_ABSOLUTE = 384138475.445311
    LIDAR_END_TIME_ABSOLUTE = 384138483.445311

    # 2. 输出模式: 'visual' (彩色投影图), 'depth' (深度NPY), 'both' (两者都输出)
    OUTPUT_MODE = 'visual'

    # 3. 指定一个独立的输出目录，用于存放本次任务的投影结果
    #    这可以防止与之前的脚本输出混淆
    TIME_SLICE_OUTPUT_DIR_NAME = "projection_of_time_slice"
    # ==============================================================================

    # --- 路径初始化 ---
    dataset_path = Path(hparams.dataset_path)
    las_path = Path(hparams.las_path)
    output_base_path = Path(hparams.las_output_path)

    # 创建一个专门的输出目录
    projection_output_path = output_base_path / TIME_SLICE_OUTPUT_DIR_NAME
    projection_output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 30)
    print("--- 开始执行特定时间片点云投影任务 ---")
    print(f"模式: 将时间范围 [{LIDAR_START_TIME_ABSOLUTE}, {LIDAR_END_TIME_ABSOLUTE}] 的点云投影到所有照片上")
    print(f"投影结果将保存到: {projection_output_path}")
    print("=" * 30)

    # --- 1. 加载LiDAR点云并根据绝对时间提取特定切片 ---
    print(f"\n[步骤 1/4] 正在从 '{las_path.name}' 加载并筛选点云...")
    try:
        with laspy.file.File(las_path, mode='r') as in_file:
            # --- 稳健性修复 START ---
            # 检查文件头中是否真的包含GPSTime范围，而不是盲目假设
            if len(in_file.header.min) > 3 and len(in_file.header.max) > 3:
                print(f"  > 调试: 文件头中找到的时间范围: [{in_file.header.min[3]:.6f}, {in_file.header.max[3]:.6f}]")
            else:
                # 如果文件头没有，就从实际的点数据中动态计算
                print("  > 调试: 文件头中未记录GPS时间范围。将从点数据中动态计算...")
                # 确保 in_file.gps_time 存在且不为空
                if hasattr(in_file, 'gps_time') and len(in_file.gps_time) > 0:
                    min_gps_time = np.min(in_file.gps_time)
                    max_gps_time = np.max(in_file.gps_time)
                    print(f"  > 调试: 从点数据中计算出的总时间范围: [{min_gps_time:.6f}, {max_gps_time:.6f}]")
                else:
                    print("  > 警告: LAS文件中找不到任何有效的GPS时间戳数据。")
            # --- 稳健性修复 END ---

            # 构建筛选掩码
            print(f"  > 正在使用指定时间范围 [{LIDAR_START_TIME_ABSOLUTE}, {LIDAR_END_TIME_ABSOLUTE}] 进行筛选...")
            time_mask = (in_file.gps_time >= LIDAR_START_TIME_ABSOLUTE) & \
                        (in_file.gps_time < LIDAR_END_TIME_ABSOLUTE)

            num_matched_points = np.sum(time_mask)
            print(f"  > 调试: 在时间范围内找到 {num_matched_points} 个匹配的点。")

            if num_matched_points == 0:
                print(
                    f"❌ 错误: 在指定的时间范围 [{LIDAR_START_TIME_ABSOLUTE}, {LIDAR_END_TIME_ABSOLUTE}] 内未找到任何点云。")
                print("  > 请检查您的时间范围是否正确，以及是否在LAS文件的总时间范围内。")
                sys.exit(1)

            # 提取点
            x = in_file.x[time_mask]
            y = in_file.y[time_mask]
            z = in_file.z[time_mask]

            # 提取颜色
            point_format = in_file.point_format
            if 'red' in point_format.lookup:
                r = (in_file.red[time_mask] / 65536.0 * 255).astype(np.uint8)
                g = (in_file.green[time_mask] / 65536.0 * 255).astype(np.uint8)
                b = (in_file.blue[time_mask] / 65536.0 * 255).astype(np.uint8)
            else:
                print("  > 警告: LAS文件中未找到颜色信息，将使用默认白色。")
                r = np.full(len(x), 255, dtype=np.uint8)
                g = r;
                b = r

            # 组合成一个Numpy数组
            points_slice_with_color = np.vstack((x, y, z, r, g, b)).T

    except Exception as e:
        print(f"❌ 错误: 加载或处理LAS文件失败: {e}")
        # 打印更详细的错误追溯信息，便于进一步调试
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"  > 点云切片加载成功！共提取 {len(points_slice_with_color)} 个点。")

    # --- 2. 加载所有全局信息 (相机位姿、坐标系等) ---
    # --- 2. 加载所有全局信息 (相机位姿、坐标系等) ---
    print("\n[步骤 2/4] 正在加载所有相机关联信息...")

    # --- 稳健性修复 START ---
    # 统一将从 hparams 传入的路径字符串转换为 Path 对象，以避免 'str' object has no attribute 'name' 错误
    infos_path = Path(hparams.infos_path)
    original_images_list_json_path = Path(hparams.original_images_list_json_path)
    # --- 稳健性修复 END ---

    try:
        # 使用转换后的 Path 对象
        root = ET.parse(infos_path).getroot()
        xml_pose = np.array([[float(p.find('Center/x').text), float(p.find('Center/y').text),
                              float(p.find('Center/z').text), float(p.find('Rotation/Omega').text),
                              float(p.find('Rotation/Phi').text), float(p.find('Rotation/Kappa').text)] for p in
                             root.findall('Block/Photogroups/Photogroup/Photo/Pose')])
        images_name = [img.text.split("\\")[-1] for img in root.findall('Block/Photogroups/Photogroup/Photo/ImagePath')]
        # 使用 .name 属性现在是安全的
        print(f"  > 成功解析 XML 文件: {infos_path.name}")
    except Exception as e:
        # 打印时也使用转换后的 Path 对象
        print(f"❌ 错误: 解析 XML 文件 '{infos_path}' 失败: {e}");
        sys.exit(1)

    try:
        # 使用转换后的 Path 对象
        with open(original_images_list_json_path, "r") as file:
            json_data = json.load(file)
            sorted_json_data = sorted(json_data, key=lambda x: x["origin_path"])
        # 使用 .name 属性现在是安全的
        print(f"  > 成功加载 JSON 文件: {original_images_list_json_path.name}")
    except Exception as e:
        # 打印时也使用转换后的 Path 对象
        print(f"❌ 错误: 加载或解析 JSON 文件 '{original_images_list_json_path}' 失败: {e}");
        sys.exit(1)

    all_process_data = []
    for i in range(len(sorted_json_data)):
        json_line = sorted_json_data[i]
        target_image_name = json_line['id'] + '.jpg'
        if target_image_name in images_name:
            index = images_name.index(target_image_name)
            path_segments = json_line['origin_path'].split('/')
            last_two_path = '/'.join(path_segments[-2:])
            all_process_data.append(
                {'index': i, 'images_name': images_name[index], 'original_image_name': last_two_path,
                 'pose': xml_pose[index, :]})
    if not all_process_data: sys.exit("❌ 错误: 未能匹配任何照片信息。")
    print(f"  > 成功加载并匹配了 {len(all_process_data)} 张照片的位姿。")

    coordinates_path = dataset_path / 'coordinates.pt'
    if not coordinates_path.exists(): sys.exit(f"❌ 错误: 找不到坐标信息文件 '{coordinates_path}'。")
    try:
        coordinate_info = torch.load(coordinates_path, weights_only=False)
        origin_drb = coordinate_info['origin_drb'].numpy()
        pose_scale_factor = coordinate_info['pose_scale_factor']
        print("  > 成功加载坐标系信息。")
    except Exception as e:
        sys.exit(f"❌ 错误: 加载坐标信息文件 '{coordinates_path}' 失败: {e}")

    # --- 3. 准备点云数据以进行投影 ---
    print("\n[步骤 3/4] 正在准备点云数据进行坐标变换...")
    points_geom = points_slice_with_color[:, :3]
    points_color = points_slice_with_color[:, 3:]

    # 这部分变换与脚本2中相同，只需执行一次
    ZYQ = torch.DoubleTensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    ZYQ_1 = torch.DoubleTensor(
        [[1, 0, 0], [0, math.cos(rad(135)), math.sin(rad(135))], [0, -math.sin(rad(135)), math.cos(rad(135))]])
    points_nerf = ZYQ.numpy() @ points_geom.T
    points_nerf = (ZYQ_1.numpy() @ points_nerf).T
    points_nerf = (points_nerf - origin_drb) / pose_scale_factor
    points_homogeneous = np.hstack((points_nerf, np.ones((points_nerf.shape[0], 1))))
    print("  > 点云已转换到 NeRF 世界坐标系。")

    # --- 4. 循环所有照片，执行投影并保存 ---
    print("\n[步骤 4/4] 开始将点云切片投影到每一张照片上...")

    files_saved_count = 0
    # 使用tqdm创建进度条
    for photo_data in tqdm(all_process_data, desc="投影到所有照片"):
        target_index = photo_data['index']

        # --- a. 获取该照片的内外参 ---
        # 确定照片属于 train 还是 val set 来找到正确的metadata路径
        # (这部分逻辑与脚本2相同)
        split_name = 'val' if target_index % int(len(all_process_data) / hparams.num_val) == 0 else 'train'
        nerf_metadata_path = dataset_path / split_name / 'metadata' / f"{target_index:06d}.pt"

        if not nerf_metadata_path.exists():
            # 使用tqdm.write来打印信息，避免破坏进度条
            tqdm.write(f"  > 警告: 照片 {target_index} 的 NeRF metadata 文件未找到，跳过。路径: {nerf_metadata_path}")
            continue

        try:
            nerf_metadata = torch.load(nerf_metadata_path, map_location='cpu')
            c2w_nerf = nerf_metadata['c2w'].numpy()
            intrinsics = nerf_metadata['intrinsics'].numpy()
            H, W = nerf_metadata['H'], nerf_metadata['W']
        except Exception as e:
            tqdm.write(f"  > 警告: 加载照片 {target_index} 的 NeRF metadata 失败: {e}，跳过。")
            continue

        # --- b. 计算投影矩阵 ---
        down_scale = hparams.down_scale
        down_scale_actual = H / int(H / down_scale)
        camera_matrix = np.array([[intrinsics[0] / down_scale_actual, 0, intrinsics[2] / down_scale_actual],
                                  [0, intrinsics[1] / down_scale_actual, intrinsics[3] / down_scale_actual],
                                  [0, 0, 1]])

        # 修正世界到相机的变换矩阵 (与脚本2中的修正完全相同)
        c2w_corrected = c2w_nerf.copy()
        c2w_corrected[:3, 1] *= -1
        c2w_corrected[:3, 2] *= -1
        c2w_4x4 = np.vstack([c2w_corrected, [0, 0, 0, 1]])
        w2c_4x4 = np.linalg.inv(c2w_4x4)
        w2c_nerf = w2c_4x4[:3, :]

        image_height = int(H / down_scale_actual)
        image_width = int(W / down_scale_actual)

        # --- c. 执行投影计算 ---
        points_cam_coord = (w2c_nerf @ points_homogeneous.T)

        mask_z = points_cam_coord[2, :] > 1e-5
        if not np.any(mask_z): continue  # 如果所有点都在相机后方，则跳过

        points_cam_coord_valid = points_cam_coord[:, mask_z]
        points_color_valid = points_color[mask_z]

        points_image_coord = (camera_matrix @ points_cam_coord_valid)
        depths = points_image_coord[2, :]
        points_image_coord = points_image_coord[:2, :] / depths

        mask_u = (points_image_coord[0, :] >= 0) & (points_image_coord[0, :] < image_width)
        mask_v = (points_image_coord[1, :] >= 0) & (points_image_coord[1, :] < image_height)
        mask_valid_uv = np.logical_and(mask_u, mask_v)

        points_uv = points_image_coord[:, mask_valid_uv].T.astype(int)
        points_color_final = points_color_valid[mask_valid_uv]
        depths_final = depths[mask_valid_uv]

        num_projected_points = len(points_uv)
        if num_projected_points == 0: continue  # 如果没有点投影到画面内，则跳过

        # --- d. 生成并保存输出 ---
        # 文件名格式： 投影到照片_{照片索引}_点数_{投影点数}.png
        output_basename = f"projected_onto_photo_{target_index:06d}_points_{num_projected_points}"

        try:
            if OUTPUT_MODE in ['visual', 'both']:
                projection_vis = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                # 使用BGR顺序，因为cv2.imwrite默认使用BGR
                projection_vis[points_uv[:, 1], points_uv[:, 0]] = points_color_final[:, ::-1]
                output_png_path = projection_output_path / f"{output_basename}.png"
                cv2.imwrite(str(output_png_path), projection_vis)
                files_saved_count += 1

            # (如果需要，也可以在这里添加深度图的保存逻辑)

        except Exception as e:
            tqdm.write(f"  > 错误: 为照片 {target_index} 保存文件时出错: {e}")

    print("\n" + "=" * 30)
    print("--- 任务执行完毕 ---")
    print(f"成功为 {files_saved_count} 张照片生成了投影图像。")
    print(f"所有结果均已保存到目录: {projection_output_path}")
    print("=" * 30)


if __name__ == '__main__':
    hparams = _get_opts()

    # 增加新的命令行参数来覆盖默认时间（如果需要的话）
    import argparse

    parser = argparse.ArgumentParser(description="Project a specific time slice of LiDAR data onto all images.")
    # 让 _get_opts 先解析
    hparams, unknown = parser.parse_known_args(args=None, namespace=hparams)

    # 添加我们自己的新参数
    parser.add_argument('--start_time', type=float, help='Absolute GPS start time for the LiDAR slice.')
    parser.add_argument('--end_time', type=float, help='Absolute GPS end time for the LiDAR slice.')

    # 再次解析，这次会解析我们自己的新参数
    hparams, unknown = parser.parse_known_args(args=None, namespace=hparams)


    # 创建一个主函数并传入参数
    def run_projection(hparams_obj, start_time_override=None, end_time_override=None):

        # 将配置移到这里，以便可以被命令行参数覆盖
        class Config:
            LIDAR_START_TIME_ABSOLUTE = 384138438.445311
            LIDAR_END_TIME_ABSOLUTE = 384138440.445311
            OUTPUT_MODE = 'visual'
            TIME_SLICE_OUTPUT_DIR_NAME = "projection_of_time_slice"

        # 如果命令行提供了时间，就使用命令行的时间
        if start_time_override:
            Config.LIDAR_START_TIME_ABSOLUTE = start_time_override
            print(f"[控制] 使用命令行指定的开始时间: {start_time_override}")
        if end_time_override:
            Config.LIDAR_END_TIME_ABSOLUTE = end_time_override
            print(f"[控制] 使用命令行指定的结束时间: {end_time_override}")

        # 将配置挂载到 hparams 对象上传递给 main 函数（这是一种简便的做法）
        hparams_obj.config = Config

        # 调用主逻辑
        main(hparams_obj)


    # 运行
    main(hparams)

#python 2-samelds-difpic.py --dataset_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output --original_images_path F:\download\SMBU2\smbu11161103\images\survey --infos_path F:\download\SMBU2\smbu11161103\AT\BlocksExchangeUndistortAT.xml --original_images_list_json_path F:\download\SMBU2\smbu11161103\images\survey\image_list.json --las_output_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output --las_path F:\download\SMBU2\SMBU1\lidars\terra_las\cloud_merged.las --num_val 10 --down_scale 4

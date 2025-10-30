
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2
from pathlib import Path
import json
from PIL import Image
import math
import torch
import os
import sys
import pickle
import datetime
import re

# --- 项目路径设置 ---
# ... (保持不变) ...
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)
from dji.opt import _get_opts


# --- 辅助函数 (保持不变) ---
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
    # --- 用户配置区域 ---
    OUTPUT_MODE = 'visual'
    # ==============================================================================

    print(f'Debug 模式: {hparams.debug}')
    down_scale = hparams.down_scale

    dataset_path = Path(hparams.dataset_path)
    original_images_path = Path(hparams.original_images_path)
    chunks_input_dir = Path(hparams.las_output_path) / 'lidar_chunks'
    projection_output_path = dataset_path / 'projection_second_chunks'
    projection_output_path.mkdir(parents=True, exist_ok=True)
    print(f"将从 '{chunks_input_dir}' 读取点云块。")
    print(f"投影结果将保存到: {projection_output_path}")

    # --- 1. 扫描 chunks 目录 ---
    print("\n正在扫描 lidar_chunks 目录...")
    chunk_files = sorted(list(chunks_input_dir.glob('points_lidar_chunks_*.pkl')))
    target_indices_from_files = []
    file_pattern = re.compile(r"points_lidar_chunks_(\d{6}).pkl")
    for f_path in chunk_files:
        match = file_pattern.match(f_path.name)
        if match: target_indices_from_files.append(int(match.group(1)))
    if not target_indices_from_files:
        print(f"❌ 错误: 在目录 '{chunks_input_dir}' 中没有找到任何 .pkl 文件。");
        sys.exit(1)
    print(f"找到 {len(target_indices_from_files)} 个对应的点云块文件，将处理以下索引:\n  {target_indices_from_files}")

    # --- 2. 加载全局信息 ---
    print("\n加载全局信息...")
    # ... (加载 XML, JSON, 匹配位姿, 加载坐标系信息的代码保持不变) ...
    try:
        root = ET.parse(hparams.infos_path).getroot();
        xml_pose = np.array([[float(p.find('Center/x').text), float(p.find('Center/y').text),
                              float(p.find('Center/z').text), float(p.find('Rotation/Omega').text),
                              float(p.find('Rotation/Phi').text), float(p.find('Rotation/Kappa').text)] for p in
                             root.findall('Block/Photogroups/Photogroup/Photo/Pose')]);
        images_name = [img.text.split("\\")[-1] for img in root.findall('Block/Photogroups/Photogroup/Photo/ImagePath')]
    except Exception as e:
        print(f"❌ 错误: 解析 XML 文件 '{hparams.infos_path}' 失败: {e}"); sys.exit(1)
    try:
        with open(hparams.original_images_list_json_path, "r") as file:
            json_data = json.load(file); sorted_json_data = sorted(json_data, key=lambda x: x["origin_path"])
    except Exception as e:
        print(f"❌ 错误: 加载或解析 JSON 文件 '{hparams.original_images_list_json_path}' 失败: {e}"); sys.exit(1)
    all_process_data = []
    for i in range(len(sorted_json_data)):
        json_line = sorted_json_data[i];
        target_image_name = json_line['id'] + '.jpg'
        if target_image_name in images_name:
            index = images_name.index(target_image_name);
            path_segments = json_line['origin_path'].split('/');
            last_two_path = '/'.join(path_segments[-2:])
            all_process_data.append(
                {'index': i, 'images_name': images_name[index], 'original_image_name': last_two_path,
                 'pose': xml_pose[index, :]})
    print(f"加载并匹配了 {len(all_process_data)} 张照片的位姿。")
    if not all_process_data: sys.exit("❌ 错误:未能匹配任何照片信息。")
    coordinates_path = dataset_path / 'coordinates.pt'
    if not coordinates_path.exists(): sys.exit(f"❌ 错误: 找不到坐标信息文件 '{coordinates_path}'。")
    try:
        coordinate_info = torch.load(coordinates_path, weights_only=False);
        origin_drb = coordinate_info['origin_drb'].numpy();
        pose_scale_factor = coordinate_info['pose_scale_factor']
    except Exception as e:
        sys.exit(f"❌ 错误: 加载坐标信息文件 '{coordinates_path}' 失败: {e}")
    print("加载所有存在的元数据以确定 train/val 分割...")
    train_meta_paths = sorted(list((dataset_path / 'train' / 'image_metadata').glob('*.pt')))
    val_meta_paths = sorted(list((dataset_path / 'val' / 'image_metadata').glob('*.pt')))
    all_existing_metadata_paths = train_meta_paths + val_meta_paths
    loaded_metadata_sorted = []
    for path in all_existing_metadata_paths:
        try:
            metadata = torch.load(path, weights_only=False)
            if 'meta_tags' in metadata and 'EXIF DateTimeOriginal' in metadata[
                'meta_tags']: loaded_metadata_sorted.append(
                {'path': path, 'time': metadata['meta_tags']['EXIF DateTimeOriginal'].values, 'index': int(path.stem)})
        except:
            continue
    if not loaded_metadata_sorted:
        print("警告: 无法加载任何元数据，将假设所有目标都在 'train' 中。")
    else:
        loaded_metadata_sorted.sort(key=lambda x: x['time'])
        print(f"已成功加载并排序 {len(loaded_metadata_sorted)} 个元数据文件。")

    # --- 3. 循环处理自动发现的索引 ---
    for target_index in target_indices_from_files:
        print(f"\n--- 开始处理目标照片索引: {target_index} ---")

        # --- a. 加载块文件 ---
        chunk_file_path = chunks_input_dir / f'points_lidar_chunks_{target_index:06d}.pkl'
        if not chunk_file_path.exists(): print(f"  ❓ 内部错误: 文件 {chunk_file_path.name} 消失？跳过。"); continue
        try:
            with open(chunk_file_path, "rb") as file:
                chunk_data = pickle.load(file)
            point_cloud_chunks = chunk_data["point_cloud_chunks"]
            chunk_time_offsets = chunk_data["chunk_time_offsets"]
            print(f"  成功加载包含 {len(point_cloud_chunks)} 个点云块的文件。")
        except Exception as e:
            print(f"  ❌ 错误: 加载块文件 {chunk_file_path.name} 失败: {e}，跳过。"); continue

        # --- b. 获取位姿和内参 ---
        photo_data = next((item for item in all_process_data if item['index'] == target_index), None)
        if photo_data is None: print(f"  ❌ 错误: 全局位姿列表中无索引 {target_index}，跳过。"); continue

        position_in_sorted_list = -1;
        for i, meta_item in enumerate(loaded_metadata_sorted):
            if meta_item['index'] == target_index: position_in_sorted_list = i; break

        split_name = 'train'  # Default
        if position_in_sorted_list != -1 and loaded_metadata_sorted:
            num_loaded_photos = len(loaded_metadata_sorted);
            val_interval = max(1, int(num_loaded_photos / max(1, hparams.num_val)))
            split_name = 'val' if position_in_sorted_list % val_interval == 0 else 'train'
        elif loaded_metadata_sorted:  # Found metadata but not this index
            print(f"  警告: 在加载的元数据中未找到索引 {target_index}，尝试默认 split_name。")
            split_name = 'val' if target_index % int(len(all_process_data) / hparams.num_val) == 0 else 'train'

        print(f"  将从 '{split_name}' 目录加载 NeRF metadata。")
        nerf_metadata_path = dataset_path / split_name / 'metadata' / f"{target_index:06d}.pt"
        if not nerf_metadata_path.exists(): print(
            f"  ❌ 错误: 找不到 NeRF metadata 文件 {nerf_metadata_path}，跳过。"); continue
        try:
            nerf_metadata = torch.load(nerf_metadata_path, map_location='cpu');
            c2w_nerf = nerf_metadata['c2w'].numpy();
            intrinsics = nerf_metadata['intrinsics'].numpy();
            H = nerf_metadata['H'];
            W = nerf_metadata['W']
        except Exception as e:
            print(f"  ❌ 错误: 加载 NeRF metadata {nerf_metadata_path} 失败: {e}，跳过。"); continue

        down_scale_actual = H / int(H / down_scale);
        camera_matrix = np.array([[intrinsics[0] / down_scale_actual, 0, intrinsics[2] / down_scale_actual],
                                  [0, intrinsics[1] / down_scale_actual, intrinsics[3] / down_scale_actual],
                                  [0, 0, 1]]);

        # ==============================================================================
        # --- FIX START: 修正世界坐标到相机坐标的变换矩阵 ---
        # 原始的 c2w (camera-to-world) 矩阵遵循 NeRF/OpenCV 的坐标系约定 (X right, Y down, Z forward)。
        # 参考脚本表明，点云数据需要一个不同的相机坐标系约定 (可能是 X right, Y up, Z backward)。
        # 我们通过翻转 c2w 矩阵的 Y 和 Z 轴来修正这个差异，然后再求逆得到 w2c 矩阵。

        # 1. 复制 c2w 矩阵以避免修改原始数据
        c2w_corrected = c2w_nerf.copy()

        # 2. 翻转 Y 和 Z 轴（对应矩阵的第2和第3列）
        c2w_corrected[:3, 1] *= -1
        c2w_corrected[:3, 2] *= -1

        # 3. 将修正后的 3x4 c2w 矩阵扩展为 4x4
        c2w_4x4 = np.vstack([c2w_corrected, [0, 0, 0, 1]])

        # 4. 求逆得到 4x4 的 w2c (world-to-camera) 矩阵
        w2c_4x4 = np.linalg.inv(c2w_4x4)

        # 5. 取其前 3 行作为最终的 3x4 变换矩阵
        w2c_nerf = w2c_4x4[:3, :]
        # --- FIX END ---
        # ==============================================================================

        image_height = int(H / down_scale_actual);
        image_width = int(W / down_scale_actual)
        photo_output_dir = projection_output_path / f"photo_{target_index:06d}";
        photo_output_dir.mkdir(parents=True, exist_ok=True)

        # --- c. 循环处理每个点云块 ---
        print(f"  开始投影 {len(point_cloud_chunks)} 个点云块...")
        files_saved_count = 0
        for chunk_idx, points_chunk in enumerate(point_cloud_chunks):
            time_offset = chunk_time_offsets[chunk_idx]
            num_points_in_chunk = len(points_chunk)

            print(f"    处理块 {chunk_idx:02d} (偏移 {time_offset:+03d}s): ", end="")

            if num_points_in_chunk == 0:
                print("点数为空，跳过。")
                continue
            print(f"包含 {num_points_in_chunk} 个点...", end="")

            points_geom = points_chunk[:, :3];
            points_color = points_chunk[:, 3:]

            # --- d. 执行投影 ---
            ZYQ = torch.DoubleTensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]]);
            ZYQ_1 = torch.DoubleTensor(
                [[1, 0, 0], [0, math.cos(rad(135)), math.sin(rad(135))], [0, -math.sin(rad(135)), math.cos(rad(135))]])
            points_nerf = ZYQ.numpy() @ points_geom.T;
            points_nerf = (ZYQ_1.numpy() @ points_nerf).T;
            points_nerf = (points_nerf - origin_drb) / pose_scale_factor
            points_homogeneous = np.hstack((points_nerf, np.ones((points_nerf.shape[0], 1))));
            points_cam_coord = (w2c_nerf @ points_homogeneous.T)

            mask_z = points_cam_coord[2, :] > 1e-5
            if not np.any(mask_z):
                # 经过修正后，这里应该不会再轻易出现所有点都在后方的情况了
                print("有效点数为0（所有点都在相机后方或过近），跳过。")
                continue

            points_cam_coord = points_cam_coord[:, mask_z];
            points_color_valid = points_color[mask_z]

            points_image_coord = (camera_matrix @ points_cam_coord);
            depths = points_image_coord[2, :];
            points_image_coord = points_image_coord[:2, :] / depths

            mask_u = (points_image_coord[0, :] >= 0) & (points_image_coord[0, :] < image_width);
            mask_v = (points_image_coord[1, :] >= 0) & (points_image_coord[1, :] < image_height);
            mask_valid_uv = np.logical_and(mask_u, mask_v)

            points_uv = points_image_coord[:, mask_valid_uv].T.astype(int);
            points_color_final = points_color_valid[mask_valid_uv];
            depths_final = depths[mask_valid_uv]

            num_projected_points = len(points_uv)
            print(f"有效投影点数: {num_projected_points}...", end="")

            if num_projected_points == 0:
                print("无有效投影点，跳过。")
                continue

            # --- e. 生成并保存输出 ---
            output_basename = f"photo_{target_index:06d}_chunk_{chunk_idx:02d}_offset_{time_offset:+03d}s"

            try:
                if OUTPUT_MODE in ['depth', 'both']:
                    large_int = 1e6;
                    depth_map = large_int * np.ones((image_height, image_width), dtype=np.float32);
                    sort_idx = np.argsort(depths_final);
                    points_uv_sorted = points_uv[sort_idx];
                    depths_final_sorted = depths_final[sort_idx];
                    depth_map[points_uv_sorted[:, 1], points_uv_sorted[:, 0]] = depths_final_sorted;
                    depth_map_metric = depth_map.copy();
                    valid_mask_metric = depth_map_metric < large_int;
                    depth_map_metric[valid_mask_metric] *= pose_scale_factor

                    output_npy_path = photo_output_dir / f"{output_basename}_depth_metric.npy"
                    print(f"正在保存 NPY: {output_npy_path.name}...", end="")
                    np.save(output_npy_path, depth_map_metric)
                    print("完成. ", end="")
                    files_saved_count += 1

                if OUTPUT_MODE in ['visual', 'both']:
                    projection_vis = np.zeros((image_height, image_width, 3), dtype=np.uint8);
                    projection_vis[points_uv[:, 1], points_uv[:, 0]] = points_color_final[:, ::-1]  # BGR

                    output_png_path = photo_output_dir / f"{output_basename}_projection.png"
                    print(f"正在保存 PNG: {output_png_path.name}...", end="")
                    cv2.imwrite(str(output_png_path), projection_vis)
                    print("完成. ", end="")
                    files_saved_count += 1

                print()

            except Exception as e:
                print(f"❌ 保存文件时出错: {e}")

        print(f"  已完成照片 {target_index} 的投影，共尝试保存 {files_saved_count} 个文件。")

    print("\n" + "=" * 25 + " 脚本执行完毕 " + "=" * 25)


if __name__ == '__main__':
    # 注意: _get_opts() 需要 dji.opt 中的定义，请确保该文件可用
    # 例如，你可以创建一个简单的 argparse 对象来测试
    class MockHparams:
        def __init__(self):
            self.dataset_path = r"F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output"
            self.original_images_path = r"F:\download\SMBU2\smbu11161103\images\survey"
            self.infos_path = r"F:\download\SMBU2\smbu11161103\AT\BlocksExchangeUndistortAT.xml"
            self.original_images_list_json_path = r"F:\download\SMBU2\smbu11161103\images\survey\image_list.json"
            self.las_output_path = r"F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output"
            self.num_val = 10
            self.down_scale = 4
            self.debug = False # or True
            # ... 其他参数 ...

    # main(MockHparams()) # 使用模拟参数进行测试
    main(_get_opts()) # 使用原始的参数加载方式









'''
python 2_convert_lidar_2_depth_color_2.py --dataset_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output --original_images_path F:\download\SMBU2\smbu11161103\images\survey --infos_path F:\download\SMBU2\smbu11161103\AT\BlocksExchangeUndistortAT.xml --original_images_list_json_path F:\download\SMBU2\smbu11161103\images\survey\image_list.json --las_output_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output --num_val 10 --start 0 --end 107 --down_scale 4 --fx 3691.094 --fy 3691.094 --cx 2755.620 --cy 1797.584 --k1 0.002214303 --k2 -0.009188852 --p1 -0.002608224 --p2 0.001401760 --k3 0.007309819



'''
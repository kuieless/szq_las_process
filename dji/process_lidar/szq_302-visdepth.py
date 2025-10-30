# # # 脚本功能：
# # # 1. (已修改) 加载 "depth_dji" 文件夹中的 .npy 深度图 (无 train/val 划分)。
# # # 2. (已修改) 将恢复了真实尺度的 .npy 深度图保存到 'depth_metric' 文件夹中 (无 train/val 划分)。
# # # 3. 保留可视化功能，生成对比图。

# # import numpy as np
# # import cv2
# # from pathlib import Path
# # from tqdm import tqdm
# # import os
# # import sys
# # import torch
# # import exifread

# # # --- 项目路径设置 (与您的脚本保持一致) ---
# # current_dir = os.path.dirname(os.path.abspath(__file__))
# # parent_dir = os.path.dirname(current_dir)
# # grandparent_dir = os.path.dirname(parent_dir)
# # if grandparent_dir not in sys.path:
# #     sys.path.append(grandparent_dir)
# # from dji.opt import _get_opts


# # # --- 辅助函数 (保持不变) ---
# # def create_color_bar(height, min_val, max_val, colormap=cv2.COLORMAP_JET):
# #     """创建一个带有刻度和文字的深度标尺图像"""
# #     bar_width = 100
# #     text_width = 150

# #     gradient = np.arange(height, 0, -1, dtype=np.float32).reshape(height, 1)
# #     gradient = (gradient / height * 255).astype(np.uint8)
# #     color_bar = cv2.applyColorMap(gradient, colormap)

# #     canvas = np.full((height, bar_width + text_width, 3), 255, dtype=np.uint8)
# #     canvas[:, :bar_width] = color_bar

# #     num_ticks = 7
# #     for i in range(num_ticks):
# #         y = int((i / (num_ticks - 1)) * (height - 1))
# #         depth_val = max_val - (i / (num_ticks - 1)) * (max_val - min_val)
# #         y_pos = min(max(y, 15), height - 15)

# #         cv2.line(canvas, (bar_width - 10, y_pos), (bar_width, y_pos), (0, 0, 0), 2)
# #         cv2.putText(canvas, f"{depth_val:.2f} m", (bar_width + 10, y_pos + 5),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# #     return canvas


# # def get_altitude_from_exif(tags):
# #     """从exifread库的tags中提取GPS高度"""
# #     alt_tag = 'GPS GPSAltitude'
# #     if alt_tag in tags:
# #         try:
# #             altitude_ratio = tags[alt_tag].values[0]
# #             altitude = altitude_ratio.num / altitude_ratio.den
# #             return f"{altitude:.2f} m"
# #         except (AttributeError, ZeroDivisionError, IndexError):
# #             return "N/A"
# #     return "N/A"


# # def main(hparams):
# #     dataset_path = Path(hparams.dataset_path) # 例如: .../output/hav

# #     # --- (核心修改) 定义两个独立的输出路径 ---
# #     vis_output_path = dataset_path / 'depth_visualized_combined'
# #     metric_output_path = dataset_path / 'depth_metric'

# #     vis_output_path.mkdir(parents=True, exist_ok=True)
# #     metric_output_path.mkdir(parents=True, exist_ok=True)
    
# #     # (!!!) (已删除) 移除了创建 train/val 子目录的逻辑
    
# #     print(f"可视化对比图将保存到: {vis_output_path}")
# #     print(f"尺度恢复后的 .npy 文件将保存到: {metric_output_path}")
# #     # --- 修改结束 ---

# #     # --- 加载坐标缩放因子 ---
# #     coordinates_path = dataset_path / 'coordinates.pt'
# #     if not coordinates_path.exists():
# #         print(f"❌ 错误: 找不到坐标信息文件 '{coordinates_path}'。")
# #         return

# #     coordinate_info = torch.load(coordinates_path)
# #     pose_scale_factor = coordinate_info['pose_scale_factor']
# #     print(f"加载的场景缩放因子 (pose_scale_factor): {pose_scale_factor:.4f}")

# #     # --- (!!!) (核心修改) 查找所有 .npy 文件 ---
# #     # (直接在 depth_dji 文件夹中查找)
# #     input_depth_dir = dataset_path / 'depth_dji'
# #     if not input_depth_dir.exists():
# #         print(f"❌ 错误: 找不到输入深度目录 '{input_depth_dir}'。")
# #         return
        
# #     all_depth_paths = sorted(list(input_depth_dir.glob('*.npy')))

# #     if not all_depth_paths:
# #         print(f"错误: 在 {input_depth_dir} 下未找到任何 .npy 文件。")
# #         return
# #     print(f"共找到 {len(all_depth_paths)} 个深度图文件需要处理。")

# #     for npy_path in tqdm(all_depth_paths, desc="正在处理深度图"):
# #         base_name = npy_path.stem
        
# #         # (!!!) (核心修改) 
# #         # parent_dir 现在是 .../output/hav (npy_path.parent.parent)
# #         # 这与您的 rgbs 和 image_metadata 路径匹配
# #         parent_dir = npy_path.parent.parent 

# #         rgb_path = parent_dir / 'rgbs' / f"{base_name}.jpg"
# #         metadata_path = parent_dir / 'image_metadata' / f"{base_name}.pt"

# #         if not rgb_path.exists():
# #             print(f"  警告: 找不到 {rgb_path}, 跳过 {base_name}")
# #             continue
# #         if not metadata_path.exists():
# #             print(f"  警告: 找不到 {metadata_path}, 跳过 {base_name}")
# #             continue

# #         # --- a. 加载原始数据 ---
# #         depth_map_normalized = np.load(npy_path)  # 这是归一化后的深度

# #         # --- b. (核心功能 1) 将深度值还原为米并保存 ---
# #         large_int = 1e6
# #         invalid_mask = (depth_map_normalized >= large_int)

# #         depth_map_meters = depth_map_normalized.copy()
# #         depth_map_meters[~invalid_mask] *= pose_scale_factor

# #         # (!!!) (核心修改) 保存恢复尺度后的 .npy 文件
# #         # (直接保存到 depth_metric 根目录)
# #         metric_npy_output_path = metric_output_path / npy_path.name
# #         np.save(metric_npy_output_path, depth_map_meters)

# #         # --- c. (核心功能 2) 使用米制单位的深度图进行可视化 ---
# #         rgb_image = cv2.imread(str(rgb_path))
# #         metadata = torch.load(metadata_path, weights_only=False)
# #         altitude = get_altitude_from_exif(metadata['meta_tags'])

# #         h, w = depth_map_meters.shape[:2]
# #         rgb_image = cv2.resize(rgb_image, (w, h))

# #         valid_depths_meters = depth_map_meters[~invalid_mask]

# #         if valid_depths_meters.size == 0:
# #             vis_depth_map = np.zeros_like(rgb_image)
# #             min_depth, max_depth = 0.0, 0.0
# #         else:
# #             min_depth = np.percentile(valid_depths_meters, 1)
# #             max_depth = np.percentile(valid_depths_meters, 99)

# #             depth_vis = depth_map_meters.copy()
# #             depth_vis[depth_vis < min_depth] = min_depth
# #             depth_vis[depth_vis > max_depth] = max_depth
# #             depth_vis[invalid_mask] = min_depth

# #             depth_vis = (depth_vis - min_depth) / max(max_depth - min_depth, 1e-8)
# #             depth_vis = (depth_vis * 255).astype(np.uint8)

# #             vis_depth_map = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
# #             vis_depth_map[invalid_mask.squeeze()] = [0, 0, 0]

# #         color_bar = create_color_bar(h, min_depth, max_depth)
# #         cv2.putText(rgb_image, f"Altitude: {altitude}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
# #         cv2.putText(vis_depth_map, f"Depth Range: {min_depth:.2f}m - {max_depth:.2f}m", (20, 40),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
# #         combined_image = np.hstack((rgb_image, vis_depth_map, color_bar))

# #         output_filename = npy_path.stem + '_comparison.png'
# #         output_filepath = vis_output_path / output_filename
# #         cv2.imwrite(str(output_filepath), combined_image)

# #     print("\n" + "=" * 25 + " 所有任务完成 " + "=" * 25)


# # if __name__ == '__main__':
# #     main(_get_opts())

# # # (!!!) (已修改) 您的示例命令是正确的
# # # python 302-visdepth.py --dataset_path /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/hav


# # 脚本功能：
# # 1. (已修改) 加载 "depth_dji" 文件夹中的 .npy 深度图 (无 train/val 划分)。
# # 2. (已修改) 将恢复了真实尺度的 .npy 深度图保存到 'depth_metric' 文件夹中 (无 train/val 划分)。
# # 3. 保留可视化功能，生成对比图。

# import numpy as np
# import cv2
# from pathlib import Path
# from tqdm import tqdm
# import os
# import sys
# import torch
# import exifread

# # --- 项目路径设置 (与您的脚本保持一致) ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# grandparent_dir = os.path.dirname(parent_dir)
# if grandparent_dir not in sys.path:
#     sys.path.append(grandparent_dir)
# from dji.opt import _get_opts


# # --- 辅助函数 (保持不变) ---
# def create_color_bar(height, min_val, max_val, colormap=cv2.COLORMAP_JET):
#     """创建一个带有刻度和文字的深度标尺图像"""
#     bar_width = 100
#     text_width = 150

#     gradient = np.arange(height, 0, -1, dtype=np.float32).reshape(height, 1)
#     gradient = (gradient / height * 255).astype(np.uint8)
#     color_bar = cv2.applyColorMap(gradient, colormap)

#     canvas = np.full((height, bar_width + text_width, 3), 255, dtype=np.uint8)
#     canvas[:, :bar_width] = color_bar

#     num_ticks = 7
#     for i in range(num_ticks):
#         y = int((i / (num_ticks - 1)) * (height - 1))
#         depth_val = max_val - (i / (num_ticks - 1)) * (max_val - min_val)
#         y_pos = min(max(y, 15), height - 15)

#         cv2.line(canvas, (bar_width - 10, y_pos), (bar_width, y_pos), (0, 0, 0), 2)
#         cv2.putText(canvas, f"{depth_val:.2f} m", (bar_width + 10, y_pos + 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

#     return canvas


# def get_altitude_from_exif(tags):
#     """从exifread库的tags中提取GPS高度"""
#     alt_tag = 'GPS GPSAltitude'
#     if alt_tag in tags:
#         try:
#             altitude_ratio = tags[alt_tag].values[0]
#             altitude = altitude_ratio.num / altitude_ratio.den
#             return f"{altitude:.2f} m"
#         except (AttributeError, ZeroDivisionError, IndexError):
#             return "N/A"
#     return "N/A"


# def main(hparams):
#     dataset_path = Path(hparams.dataset_path) # 例如: .../output/hav

#     # --- (核心修改) 定义两个独立的输出路径 ---
#     vis_output_path = dataset_path / 'depth_visualized_combined'
#     metric_output_path = dataset_path / 'depth_metric'

#     vis_output_path.mkdir(parents=True, exist_ok=True)
#     metric_output_path.mkdir(parents=True, exist_ok=True)
    
#     # (!!!) (已删除) 移除了创建 train/val 子目录的逻辑
    
#     print(f"可视化对比图将保存到: {vis_output_path}")
#     print(f"尺度恢复后的 .npy 文件将保存到: {metric_output_path}")
#     # --- 修改结束 ---

#     # --- 加载坐标缩放因子 ---
#     coordinates_path = dataset_path / 'coordinates.pt'
#     if not coordinates_path.exists():
#         print(f"❌ 错误: 找不到坐标信息文件 '{coordinates_path}'。")
#         return

#     coordinate_info = torch.load(coordinates_path)
#     pose_scale_factor = coordinate_info['pose_scale_factor']
#     print(f"加载的场景缩放因子 (pose_scale_factor): {pose_scale_factor:.4f}")

#     # --- (!!!) (核心修改) 查找所有 .npy 文件 ---
#     # (直接在 depth_dji 文件夹中查找)
#     input_depth_dir = dataset_path / 'depth_dji'
#     if not input_depth_dir.exists():
#         print(f"❌ 错误: 找不到输入深度目录 '{input_depth_dir}'。")
#         return
        
#     all_depth_paths = sorted(list(input_depth_dir.glob('*.npy')))

#     if not all_depth_paths:
#         # (!!!) (已修改) 错误信息现在会指向正确的目录
#         print(f"错误: 在 {input_depth_dir} 下未找到任何 .npy 文件。")
#         return
#     print(f"共找到 {len(all_depth_paths)} 个深度图文件需要处理。")

#     for npy_path in tqdm(all_depth_paths, desc="正在处理深度图"):
#         base_name = npy_path.stem
        
#         # (!!!) (核心修改) 
#         # parent_dir 现在是 .../output/hav (npy_path.parent.parent)
#         # 这与您的 rgbs 和 image_metadata 路径匹配
#         parent_dir = npy_path.parent.parent 

#         rgb_path = parent_dir / 'rgbs' / f"{base_name}.jpg"
#         metadata_path = parent_dir / 'image_metadata' / f"{base_name}.pt"

#         if not rgb_path.exists():
#             print(f"  警告: 找不到 {rgb_path}, 跳过 {base_name}")
#             continue
#         if not metadata_path.exists():
#             print(f"  警告: 找不到 {metadata_path}, 跳过 {base_name}")
#             continue

#         # --- a. 加载原始数据 ---
#         # (!!!) (修复) 确保即使加载失败也能继续
#         try:
#             depth_map_normalized = np.load(npy_path)  # 这是归一化后的深度
#         except Exception as e:
#             print(f"  警告: 加载 .npy 文件 {npy_path} 失败: {e}。跳过。")
#             continue

#         # --- b. (核心功能 1) 将深度值还原为米并保存 ---
#         large_int = 1e6
#         invalid_mask = (depth_map_normalized >= large_int)

#         depth_map_meters = depth_map_normalized.copy()
#         depth_map_meters[~invalid_mask] *= pose_scale_factor

#         # (!!!) (核心修改) 保存恢复尺度后的 .npy 文件
#         # (直接保存到 depth_metric 根目录)
#         metric_npy_output_path = metric_output_path / npy_path.name
#         np.save(metric_npy_output_path, depth_map_meters)

#         # --- c. (核心功能 2) 使用米制单位的深度图进行可视化 ---
#         rgb_image = cv2.imread(str(rgb_path))
        
#         # (!!!) (修复) 修复旧版 PyTorch 的 'weights_only' 错误
#         try:
#             metadata = torch.load(metadata_path, weights_only=False)
#         except TypeError:
#              metadata = torch.load(metadata_path) # 兼容旧版
             
#         altitude = get_altitude_from_exif(metadata['meta_tags'])

#         h, w = depth_map_meters.shape[:2]
        
#         # (!!!) (修复) 检查 rgb_image 是否加载成功
#         if rgb_image is None:
#             print(f"  警告: 加载 RGB 图像 {rgb_path} 失败。跳过。")
#             continue
            
#         rgb_image = cv2.resize(rgb_image, (w, h))

#         valid_depths_meters = depth_map_meters[~invalid_mask]

#         if valid_depths_meters.size == 0:
#             vis_depth_map = np.zeros_like(rgb_image)
#             min_depth, max_depth = 0.0, 0.0
#         else:
#             min_depth = np.percentile(valid_depths_meters, 1)
#             max_depth = np.percentile(valid_depths_meters, 99)

#             depth_vis = depth_map_meters.copy()
#             depth_vis[depth_vis < min_depth] = min_depth
#             depth_vis[depth_vis > max_depth] = max_depth
#             depth_vis[invalid_mask] = min_depth

#             depth_vis = (depth_vis - min_depth) / max(max_depth - min_depth, 1e-8)
#             depth_vis = (depth_vis * 255).astype(np.uint8)

#             vis_depth_map = cv2.applyColorMap(depth_vis.squeeze(), cv2.COLORMAP_JET)
#             vis_depth_map[invalid_mask.squeeze()] = [0, 0, 0]

#         color_bar = create_color_bar(h, min_depth, max_depth)
#         cv2.putText(rgb_image, f"Altitude: {altitude}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
#         cv2.putText(vis_depth_map, f"Depth Range: {min_depth:.2f}m - {max_depth:.2f}m", (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
#         combined_image = np.hstack((rgb_image, vis_depth_map, color_bar))

#         output_filename = npy_path.stem + '_comparison.png'
#         output_filepath = vis_output_path / output_filename
#         cv2.imwrite(str(output_filepath), combined_image)

#     print("\n" + "=" * 25 + " 所有任务完成 " + "=" * 25)


# if __name__ == '__main__':
#     main(_get_opts())

# 脚本功能：
# 1. (已修改) 加载 "depth_dji" 文件夹中的 .npy 深度图 (无 train/val 划分)。
# 2. (已修改) 将恢复了真实尺度的 .npy 深度图保存到 'depth_metric' 文件夹中 (无 train/val 划分)。
# 3. 保留可视化功能，生成对比图。

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import os
import sys
import torch
import exifread

# --- 项目路径设置 (与您的脚本保持一致) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)
from dji.opt import _get_opts


# --- 辅助函数 (保持不变) ---
def create_color_bar(height, min_val, max_val, colormap=cv2.COLORMAP_JET):
    """创建一个带有刻度和文字的深度标尺图像"""
    bar_width = 100
    text_width = 150

    gradient = np.arange(height, 0, -1, dtype=np.float32).reshape(height, 1)
    gradient = (gradient / height * 255).astype(np.uint8)
    color_bar = cv2.applyColorMap(gradient, colormap)

    canvas = np.full((height, bar_width + text_width, 3), 255, dtype=np.uint8)
    canvas[:, :bar_width] = color_bar

    num_ticks = 7
    for i in range(num_ticks):
        y = int((i / (num_ticks - 1)) * (height - 1))
        depth_val = max_val - (i / (num_ticks - 1)) * (max_val - min_val)
        y_pos = min(max(y, 15), height - 15)

        cv2.line(canvas, (bar_width - 10, y_pos), (bar_width, y_pos), (0, 0, 0), 2)
        cv2.putText(canvas, f"{depth_val:.2f} m", (bar_width + 10, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return canvas


def get_altitude_from_exif(tags):
    """从exifread库的tags中提取GPS高度"""
    alt_tag = 'GPS GPSAltitude'
    if alt_tag in tags:
        try:
            altitude_ratio = tags[alt_tag].values[0]
            altitude = altitude_ratio.num / altitude_ratio.den
            return f"{altitude:.2f} m"
        except (AttributeError, ZeroDivisionError, IndexError):
            return "N/A"
    return "N/A"


def main(hparams):
    dataset_path = Path(hparams.dataset_path) # 例如: .../output/hav

    # --- (核心修改) 定义两个独立的输出路径 ---
    vis_output_path = dataset_path / 'depth_visualized_combined'
    metric_output_path = dataset_path / 'depth_metric'

    vis_output_path.mkdir(parents=True, exist_ok=True)
    metric_output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"可视化对比图将保存到: {vis_output_path}")
    print(f"尺度恢复后的 .npy 文件将保存到: {metric_output_path}")
    # --- 修改结束 ---

    # --- 加载坐标缩放因子 ---
    coordinates_path = dataset_path / 'coordinates.pt'
    if not coordinates_path.exists():
        print(f"❌ 错误: 找不到坐标信息文件 '{coordinates_path}'。")
        return

    coordinate_info = torch.load(coordinates_path)
    pose_scale_factor = coordinate_info['pose_scale_factor']
    print(f"加载的场景缩放因子 (pose_scale_factor): {pose_scale_factor:.4f}")

    # --- (!!!) (核心修改) 查找所有 .npy 文件 ---
    # (直接在 depth_dji 文件夹中查找)
    input_depth_dir = dataset_path / 'depth_dji'
    if not input_depth_dir.exists():
        print(f"❌ 错误: 找不到输入深度目录 '{input_depth_dir}'。")
        return
        
    all_depth_paths = sorted(list(input_depth_dir.glob('*.npy')))

    if not all_depth_paths:
        # (!!!) (已修改) 错误信息现在会指向正确的目录
        print(f"错误: 在 {input_depth_dir} 下未找到任何 .npy 文件。")
        return
    print(f"共找到 {len(all_depth_paths)} 个深度图文件需要处理。")

    for npy_path in tqdm(all_depth_paths, desc="正在处理深度图"):
        base_name = npy_path.stem
        
        # (!!!) (核心修改) 
        # parent_dir 现在是 .../output/hav (npy_path.parent.parent)
        # 这与您的 rgbs 和 image_metadata 路径匹配
        parent_dir = npy_path.parent.parent 

        rgb_path = parent_dir / 'rgbs' / f"{base_name}.jpg"
        metadata_path = parent_dir / 'image_metadata' / f"{base_name}.pt"

        if not rgb_path.exists():
            print(f"  警告: 找不到 {rgb_path}, 跳过 {base_name}")
            continue
        if not metadata_path.exists():
            print(f"  警告: 找不到 {metadata_path}, 跳过 {base_name}")
            continue

        # --- a. 加载原始数据 ---
        try:
            # (!!!) (语法修复) 确保这行没有非法字符
            depth_map_normalized = np.load(npy_path) # 这是归一化后的深度
        except Exception as e:
            print(f"  警告: 加载 .npy 文件 {npy_path} 失败: {e}。跳过。")
            continue

        # --- b. (核心功能 1) 将深度值还原为米并保存 ---
        large_int = 1e6
        invalid_mask = (depth_map_normalized >= large_int)

        depth_map_meters = depth_map_normalized.copy()
        depth_map_meters[~invalid_mask] *= pose_scale_factor

        # (!!!) (核心修改) 保存恢复尺度后的 .npy 文件
        metric_npy_output_path = metric_output_path / npy_path.name
        np.save(metric_npy_output_path, depth_map_meters)

        # --- c. (核心功能 2) 使用米制单位的深度图进行可视化 ---
        rgb_image = cv2.imread(str(rgb_path))
        
        if rgb_image is None:
            print(f"  警告: 加载 RGB 图像 {rgb_path} 失败。跳过。")
            continue
        
        try:
            # 兼容新版 torch.load
            metadata = torch.load(metadata_path, weights_only=False)
        except TypeError:
             # 兼容旧版 torch.load (移除 weights_only)
             metadata = torch.load(metadata_path)
             
        altitude = get_altitude_from_exif(metadata['meta_tags'])

        h, w = depth_map_meters.shape[:2]
        rgb_image = cv2.resize(rgb_image, (w, h))

        valid_depths_meters = depth_map_meters[~invalid_mask]

        if valid_depths_meters.size == 0:
            vis_depth_map = np.zeros_like(rgb_image)
            min_depth, max_depth = 0.0, 0.0
        else:
            min_depth = np.percentile(valid_depths_meters, 1)
            max_depth = np.percentile(valid_depths_meters, 99)

            depth_vis = depth_map_meters.copy()
            depth_vis[depth_vis < min_depth] = min_depth
            depth_vis[depth_vis > max_depth] = max_depth
            depth_vis[invalid_mask] = min_depth

            depth_vis = (depth_vis - min_depth) / max(max_depth - min_depth, 1e-8)
            depth_vis = (depth_vis * 255).astype(np.uint8)

            vis_depth_map = cv2.applyColorMap(depth_vis.squeeze(), cv2.COLORMAP_JET)
            vis_depth_map[invalid_mask.squeeze()] = [0, 0, 0]

        color_bar = create_color_bar(h, min_depth, max_depth)
        cv2.putText(rgb_image, f"Altitude: {altitude}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(vis_depth_map, f"Depth Range: {min_depth:.2f}m - {max_depth:.2f}m", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        combined_image = np.hstack((rgb_image, vis_depth_map, color_bar))

        output_filename = npy_path.stem + '_comparison.png'
        output_filepath = vis_output_path / output_filename
        cv2.imwrite(str(output_filepath), combined_image)

    print("\n" + "=" * 25 + " 所有任务完成 " + "=" * 25)


if __name__ == '__main__':
    main(_get_opts())
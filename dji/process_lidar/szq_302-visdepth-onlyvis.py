import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import os
import sys
import torch
import exifread
import argparse # 使用标准库 argparse

# --- 辅助函数 (从您的脚本中保留) ---

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


def main(args):
    # --- 1. 设置路径 ---
    # 输入的 .npy 文件夹 (例如: .../output/hav/depth_metric)
    input_depth_dir = Path(args.input_dir)
    
    # 输出的可视化文件夹 (例如: .../output/hav/visualizations)
    vis_output_path = Path(args.output_dir)
    vis_output_path.mkdir(parents=True, exist_ok=True)

    # 自动推断 RGB 和 Metadata 文件夹的路径
    # 假设它们与 input_dir 共享一个父目录 (例如: .../output/hav)
    common_parent_dir = input_depth_dir.parent
    rgb_dir = common_parent_dir / 'rgbs'
    metadata_dir = common_parent_dir / 'image_metadata'

    print(f"米制 .npy 输入目录: {input_depth_dir}")
    print(f"RGB 目录: {rgb_dir}")
    print(f"Metadata 目录: {metadata_dir}")
    print(f"可视化输出目录: {vis_output_path}")

    # --- 2. 查找所有 .npy 文件 ---
    all_depth_paths = sorted(list(input_depth_dir.glob('*.npy')))
    if not all_depth_paths:
        print(f"错误: 在 {input_depth_dir} 下未找到任何 .npy 文件。")
        return
    print(f"共找到 {len(all_depth_paths)} 个深度图文件需要可视化。")

    # --- 3. 循环处理 (核心可视化逻辑) ---
    for npy_path in tqdm(all_depth_paths, desc="正在生成可视化图"):
        base_name = npy_path.stem
        
        rgb_path = rgb_dir / f"{base_name}.jpg"
        metadata_path = metadata_dir / f"{base_name}.pt"

        if not rgb_path.exists():
            print(f"  警告: 找不到 {rgb_path}, 跳过 {base_name}")
            continue
        if not metadata_path.exists():
            print(f"  警告: 找不到 {metadata_path}, 跳过 {base_name}")
            continue

        # --- a. 加载数据 ---
        try:
            # 直接加载 *米制* 深度图 (因为我们是可视化脚本)
            depth_map_meters = np.load(npy_path)
        except Exception as e:
            print(f"  警告: 加载 .npy 文件 {npy_path} 失败: {e}。跳过。")
            continue
        
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

        # --- b. 开始可视化 ---
        h, w = depth_map_meters.shape[:2]
        rgb_image = cv2.resize(rgb_image, (w, h))

        # (保留了原始脚本中的哨兵值逻辑)
        large_int = 1e6 
        # (假设米制 .npy 中的无效值仍为 1e6)
        invalid_mask = (depth_map_meters >= large_int)
        
        valid_depths_meters = depth_map_meters[~invalid_mask]

        if valid_depths_meters.size == 0:
            vis_depth_map = np.zeros_like(rgb_image)
            min_depth, max_depth = 0.0, 0.0
        else:
            # (保留了原始脚本中的百分位裁剪逻辑)
            min_depth = np.percentile(valid_depths_meters, 1)
            max_depth = np.percentile(valid_depths_meters, 99)

            depth_vis = depth_map_meters.copy()
            depth_vis[depth_vis < min_depth] = min_depth
            depth_vis[depth_vis > max_depth] = max_depth
            depth_vis[invalid_mask] = min_depth

            depth_vis = (depth_vis - min_depth) / max(max_depth - min_depth, 1e-8)
            depth_vis = (depth_vis * 255).astype(np.uint8)

            vis_depth_map = cv2.applyColorMap(depth_vis.squeeze(), cv2.COLORMAP_JET)
            vis_depth_map[invalid_mask.squeeze()] = [0, 0, 0] # 无效区域设为黑色

        # --- c. 组合并保存图像 ---
        color_bar = create_color_bar(h, min_depth, max_depth)
        cv2.putText(rgb_image, f"Altitude: {altitude}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(vis_depth_map, f"Depth Range: {min_depth:.2f}m - {max_depth:.2f}m", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        combined_image = np.hstack((rgb_image, vis_depth_map, color_bar))

        output_filename = npy_path.stem + '_comparison.png'
        output_filepath = vis_output_path / output_filename
        cv2.imwrite(str(output_filepath), combined_image)

    print("\n" + "=" * 25 + " 可视化完成 " + "=" * 25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从米制 .npy 文件生成深度图可视化")
    
    parser.add_argument(
        "-i", "--input_dir", 
        type=str, 
        required=True,
        help="包含米制 .npy 深度文件的输入目录 (例如: .../output/hav/depth_metric)"
    )
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, 
        required=True,
        help="用于保存输出可视化 PNG 图像的目录 (例如: .../output/hav/visualizations)"
    )
    
    args = parser.parse_args()
    main(args)

    # 激活您的环境 (如果需要)
# source ...
'''

# 运行可视化脚本
python szq_302-visdepth-onlyvis.py \
    -i /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/hav/depth_gt_hav \
    -o /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/hav/depth_gt_hav/vis


'''
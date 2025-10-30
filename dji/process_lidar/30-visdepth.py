# 脚本功能：
# 1. 自动查找指定数据集路径下的所有深度图 .npy 文件。
# 2. 逐一加载深度图数据。
# 3. 将原始的深度值（米）转换为彩色的、人类可读的可视化图像。
# 4. 将可视化后的深度图保存为 .png 文件，以便检查。

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import os
import sys

# --- 项目路径设置 (与您的脚本保持一致) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)
from dji.opt import _get_opts

def main(hparams):
    dataset_path = Path(hparams.dataset_path)
    
    # --- 新增配置：定义可视化结果的输出路径 ---
    # 默认保存在数据集路径下的 "depth_visualized" 文件夹中
    output_path = dataset_path / 'depth_visualized'
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"可视化结果将保存到: {output_path}")

    # --- 1. 查找所有需要可视化的 .npy 深度图文件 ---
    train_depth_paths = sorted(list((dataset_path / 'train' / 'depth_dji').glob('*.npy')))
    val_depth_paths = sorted(list((dataset_path / 'val' / 'depth_dji').glob('*.npy')))
    all_depth_paths = train_depth_paths + val_depth_paths

    if not all_depth_paths:
        print(f"错误: 在 {dataset_path} 下的 'train/depth_dji' 或 'val/depth_dji' 文件夹中没有找到任何 .npy 文件。")
        return

    print(f"共找到 {len(all_depth_paths)} 个深度图文件需要处理。")

    # --- 2. 循环处理每个深度图文件 ---
    for npy_path in tqdm(all_depth_paths, desc="Visualizing depth maps"):
        # 加载深度数据
        depth_map = np.load(npy_path)

        # 您的脚本中使用 1e6 作为无效值的标记
        large_int = 1e6
        
        # 创建一个掩码，标记出无效深度值的位置
        invalid_mask = (depth_map >= large_int)
        
        # 从有效深度值中找到最大和最小值以进行归一化
        valid_depths = depth_map[~invalid_mask]

        if valid_depths.size == 0:
            # 如果整张图都没有有效的深度值，则创建一张纯黑色的图
            print(f"警告: 文件 {npy_path.name} 不包含任何有效的深度点，将生成一张黑色图像。")
            vis_depth_map = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
        else:
            # 为了更好的可视化效果，我们通常会截取一个百分比范围，以忽略极端离群值
            min_depth = np.percentile(valid_depths, 1)
            max_depth = np.percentile(valid_depths, 99)
            
            # 创建一个新的数组用于可视化，避免修改原始数据
            depth_vis = depth_map.copy()

            # 将深度值裁剪到我们计算的范围内
            depth_vis[depth_vis < min_depth] = min_depth
            depth_vis[depth_vis > max_depth] = max_depth
            
            # 将无效值暂时也设置为最小值，以便进行归一化
            depth_vis[invalid_mask] = min_depth
            
            # 线性归一化到 0-1 范围
            depth_vis = (depth_vis - min_depth) / max(max_depth - min_depth, 1e-8)
            
            # 翻转颜色（通常近处为暖色，远处为冷色），并缩放到 0-255
            depth_vis = (depth_vis * 255).astype(np.uint8)
            
            # 应用伪彩色映射
            # COLORMAP_JET 和 COLORMAP_INFERNO 是常用的选择
            vis_depth_map = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            
            # 最后，将之前标记为无效的区域设置为一个醒目的颜色，比如白色或黑色
            vis_depth_map[invalid_mask.squeeze()] = [0, 0, 0] # 设置为黑色

        # --- 3. 保存可视化图像 ---
        output_filename = npy_path.stem + '.png'
        output_filepath = output_path / output_filename
        cv2.imwrite(str(output_filepath), vis_depth_map)

    print("\n" + "=" * 25 + " 可视化完成 " + "=" * 25)
    print("所有深度图的可视化结果均已保存。")
    print("=" * 62)

if __name__ == '__main__':
    main(_get_opts())
#python 30-visdepth.py --dataset_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output
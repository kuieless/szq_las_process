

# import numpy as np
# import os
# import sys
# import glob
# import argparse
# import subprocess

# # --- (可选) 自动安装依赖 ---
# try:
#     import matplotlib.pyplot as plt
#     import matplotlib.colors
# except ImportError:
#     print("未检测到 matplotlib，正在尝试自动安装...")
#     try:
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
#         import matplotlib.pyplot as plt
#         import matplotlib.colors
#         print("matplotlib 安装成功。")
#     except Exception as e:
#         print(f"自动安装失败: {e}\n请手动运行 'pip install matplotlib' 后再试。")
#         sys.exit(1)

# def process_and_visualize_depth(lidar_path, mesh_path, output_dir, threshold, max_depth, no_viz):
#     """
#     加载、处理并可视化单个激光雷达和Mesh深度图对。
#     """
#     base_name = os.path.basename(lidar_path).replace('.npy', '')
#     print(f"\n--- Processing: {base_name} ---")

#     # --- 1. 数据加载 ---
#     try:
#         lidar_depth = np.load(lidar_path)
#         mesh_depth = np.load(mesh_path)
#     except FileNotFoundError as e:
#         print(f"错误：文件未找到！ {e}")
#         return

#     if lidar_depth.shape != mesh_depth.shape:
#         print(f"错误：文件 {base_name} 的尺寸不匹配，跳过处理。")
#         return

#     # --- 2. 核心筛选逻辑 ---
#     lidar_valid_mask = (lidar_depth > 0) & (lidar_depth < max_depth)
#     mesh_valid_mask = (mesh_depth > 0) & (mesh_depth < max_depth)
#     both_valid_mask = lidar_valid_mask & mesh_valid_mask
#     see_through_condition = lidar_depth[both_valid_mask] > (mesh_depth[both_valid_mask] + threshold)
    
#     final_removal_mask = np.zeros_like(lidar_depth, dtype=bool)
#     final_removal_mask[both_valid_mask] = see_through_condition
#     num_filtered_points = np.sum(final_removal_mask)
    
#     # --- 3. 生成并保存清理后的深度图 ---
#     lidar_depth_cleaned = lidar_depth.copy()
#     lidar_depth_cleaned[~lidar_valid_mask] = 0
#     lidar_depth_cleaned[final_removal_mask] = 0
    
#     output_npy_path = os.path.join(output_dir, f"{base_name}_cleaned.npy")
#     np.save(output_npy_path, lidar_depth_cleaned)
#     print(f"处理完成: 找到 {num_filtered_points} 个透视点。")
#     print(f"清理后的深度图已保存到: {output_npy_path}")

#     # --- 4. 可视化 (如果需要) ---
#     if no_viz:
#         return

#     # 可视化准备
#     INVALID_COLOR = 'darkgray'
#     cmap_viridis = plt.get_cmap('viridis').copy()
#     cmap_viridis.set_bad(color=INVALID_COLOR)
#     cmap_hot = plt.get_cmap('hot').copy()
#     cmap_hot.set_bad(color=INVALID_COLOR)

#     lidar_depth_viz = lidar_depth.copy()
#     lidar_depth_viz[~lidar_valid_mask] = np.nan
#     mesh_depth_viz = mesh_depth.copy()
#     mesh_depth_viz[~mesh_valid_mask] = np.nan
#     removed_points_viz = np.full_like(lidar_depth, np.nan, dtype=np.float64)
#     removed_points_viz[final_removal_mask] = lidar_depth[final_removal_mask]
#     lidar_depth_cleaned_viz = lidar_depth_cleaned.copy().astype(float)
#     lidar_depth_cleaned_viz[lidar_depth_cleaned_viz == 0] = np.nan

#     valid_original_depth = lidar_depth[lidar_valid_mask]
#     vmax = np.percentile(valid_original_depth, 98) if valid_original_depth.size > 0 else max_depth

#     # 定义绘图数据
#     plot_titles = [
#         '1. Original LiDAR Depth', '2. Mesh Reference Depth',
#         f'3. Removed {num_filtered_points} See-Through Points', '4. Final Cleaned Depth Map'
#     ]
#     plot_data = [lidar_depth_viz, mesh_depth_viz, removed_points_viz, lidar_depth_cleaned_viz]
#     plot_cmaps = [cmap_viridis, cmap_viridis, cmap_hot, cmap_viridis]

#     # 生成并保存单张大图
#     for i in range(4):
#         fig_ind, ax_ind = plt.subplots(figsize=(12, 9))
#         im = ax_ind.imshow(plot_data[i], cmap=plot_cmaps[i], vmax=vmax)
#         ax_ind.set_title(plot_titles[i], fontsize=16)
#         ax_ind.set_facecolor(INVALID_COLOR)
#         cbar = fig_ind.colorbar(im, orientation='vertical', label='Depth (m)', pad=0.02)
        
#         simple_title = plot_titles[i].split(' ')[1]
#         output_path = os.path.join(output_dir, f"{base_name}_{i+1}_{simple_title}.png")
#         plt.savefig(output_path, dpi=200, bbox_inches='tight')
#         plt.close(fig_ind)
    
#     print(f"可视化结果已保存到: {output_dir}")

# def main():
#     parser = argparse.ArgumentParser(description="清理激光雷达深度图中的透视点，并生成可视化结果。")
    
#     # 输入参数 (文件或目录)
#     parser.add_argument('-l', '--lidar', type=str, help="单个激光雷达深度图文件路径 (.npy)。")
#     parser.add_argument('-m', '--mesh', type=str, help="单个Mesh深度图文件路径 (.npy)。")
#     parser.add_argument('--lidar_dir', type=str, help="包含激光雷达深度图的目录。")
#     parser.add_argument('--mesh_dir', type=str, help="包含Mesh深度图的目录。")
    
#     # 输出参数
#     parser.add_argument('-o', '--output_dir', type=str, default="./results", help="保存结果的目录。")
    
#     # 算法参数
#     parser.add_argument('-t', '--threshold', type=float, default=3.0, help="判断为透视点的深度差异阈值 (米)。")
#     parser.add_argument('--max_depth', type=float, default=1000.0, help="最大有效深度 (米)。")
    
#     # 功能开关
#     parser.add_argument('--no_viz', action='store_true', help="如果设置，则不生成可视化图片，只保存清理后的.npy文件。")

#     args = parser.parse_args()

#     # 创建输出目录
#     os.makedirs(args.output_dir, exist_ok=True)

#     # 判断是处理单个文件还是整个目录
#     if args.lidar and args.mesh:
#         process_and_visualize_depth(args.lidar, args.mesh, args.output_dir, args.threshold, args.max_depth, args.no_viz)
#     elif args.lidar_dir and args.mesh_dir:
#         lidar_files = sorted(glob.glob(os.path.join(args.lidar_dir, '*.npy')))
#         if not lidar_files:
#             print(f"错误: 在目录 {args.lidar_dir} 中未找到任何 .npy 文件。")
#             return
            
#         for lidar_path in lidar_files:
#             base_name = os.path.basename(lidar_path)
#             mesh_path = os.path.join(args.mesh_dir, base_name)
#             if os.path.exists(mesh_path):
#                 process_and_visualize_depth(lidar_path, mesh_path, args.output_dir, args.threshold, args.max_depth, args.no_viz)
#             else:
#                 print(f"警告: 找不到与 {base_name} 匹配的Mesh深度图，跳过。")
#     else:
#         print("错误: 请提供一对输入文件 (--lidar 和 --mesh) 或一对输入目录 (--lidar_dir 和 --mesh_dir)。")
#         parser.print_help()

# if __name__ == '__main__':
#     main()

# '''
# python compare-lidar-mesh-depth.py \
#     -l /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_dji/000010.npy \
#     -m /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_mesh/000010.npy \
#     -o ./my_single_file_results \
#     -t 3.0

# python compare-lidar-mesh-depth.py \
#     --lidar_dir /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_dji \
#     --mesh_dir /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_mesh \
#     -o ./my_batch_results \
#     -t 3.0

# '''


#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # 文件名: clean_depth.py

# import numpy as np
# import os
# import sys
# import glob
# import argparse
# import subprocess

# # --- (可选) 自动安装依赖 ---
# try:
#     import matplotlib.pyplot as plt
#     import matplotlib.colors
# except ImportError:
#     print("未检测到 matplotlib，正在尝试自动安装...")
#     try:
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
#         import matplotlib.pyplot as plt
#         import matplotlib.colors
#         print("matplotlib 安装成功。")
#     except Exception as e:
#         print(f"自动安装失败: {e}\n请手动运行 'pip install matplotlib' 后再试。")
#         sys.exit(1)

# def process_and_visualize_depth(lidar_path, mesh_path, output_dir, threshold, max_depth, no_viz, base_name):
#     """
#     加载、处理并可视化单个激光雷达和Mesh深度图对。
#     """
    
#     print(f"  - 正在处理: {base_name}.npy")

#     # --- 1. 数据加载 ---
#     try:
#         lidar_depth = np.load(lidar_path)
#         mesh_depth = np.load(mesh_path)
#     except FileNotFoundError as e:
#         print(f"    错误：文件未找到！ {e}")
#         return
#     except Exception as e:
#         print(f"    错误：加载 {base_name}.npy 时出错: {e}")
#         return

#     if lidar_depth.shape != mesh_depth.shape:
#         print(f"    错误：文件 {base_name} 的尺寸不匹配 ({lidar_depth.shape} vs {mesh_depth.shape})，跳过处理。")
#         return

#     # --- 2. 核心筛选逻辑 ---
    
#     # 找出 Lidar 和 Mesh 中“有效”的区域 (大于0 且 小于最大深度)
#     # 极大值 (如 1,000,000) 会在这里被 (lidar_depth < max_depth) 过滤掉
#     lidar_valid_mask = (lidar_depth > 0) & (lidar_depth < max_depth)
#     mesh_valid_mask = (mesh_depth > 0) & (mesh_depth < max_depth)
    
#     # 两个深度图 *同时* 有效的区域
#     both_valid_mask = lidar_valid_mask & mesh_valid_mask
    
#     # 在 "同时有效" 区域内，找出 Lidar > Mesh+阈值 的 "透视点"
#     see_through_condition = np.zeros_like(lidar_depth, dtype=bool)
#     if np.any(both_valid_mask): # 仅在有有效区域时才计算
#         see_through_condition[both_valid_mask] = lidar_depth[both_valid_mask] > (mesh_depth[both_valid_mask] + threshold)
    
#     num_filtered_points = np.sum(see_through_condition)
    
#     # --- 3. 生成并保存清理后的深度图 (关键) ---
    
#     # 1. 复制一份原始Lidar数据
#     lidar_depth_cleaned = lidar_depth.copy()
    
#     # 2. 建立一个 "全无效" 掩码，找出所有 *应该* 被设为0的点:
#     #    - Lidar 自身无效 (0 或 >= max_depth)
#     #    - Mesh 对应点无效 (0 或 >= max_depth)
#     #    - 被识别为 "透视" 的点
#     #    (使用 | 或逻辑 "或" 运算)
    
#     invalid_points_mask = (~lidar_valid_mask) | (~mesh_valid_mask) | see_through_condition
    
#     # 3. 将所有这些点在清理版中设为 0
#     lidar_depth_cleaned[invalid_points_mask] = 0
    
#     # 4. 保存 (保存为原始名称，以便GT使用)
#     output_npy_path = os.path.join(output_dir, f"{base_name}.npy") 
#     np.save(output_npy_path, lidar_depth_cleaned)
    
#     print(f"    处理完成: 找到 {num_filtered_points} 个透视点。")
#     print(f"    清理后的 GT (npy) 已保存到: {output_npy_path}")

#     # --- 4. 可视化 (如果需要) ---
#     if no_viz:
#         return

#     # 为可视化创建输出子目录 (避免 .npy 和 .png 混在一起)
#     viz_output_dir = os.path.join(output_dir, "visualizations")
#     os.makedirs(viz_output_dir, exist_ok=True)

#     # 可视化准备
#     INVALID_COLOR = 'darkgray'
#     cmap_viridis = plt.get_cmap('viridis').copy()
#     cmap_viridis.set_bad(color=INVALID_COLOR)
#     cmap_hot = plt.get_cmap('hot').copy()
#     cmap_hot.set_bad(color=INVALID_COLOR)

#     # 准备用于可视化的数据 (将 0 和无效值设为 NaN 以便绘图)
#     lidar_depth_viz = lidar_depth.copy()
#     lidar_depth_viz[~lidar_valid_mask] = np.nan # 原Lidar无效点
    
#     mesh_depth_viz = mesh_depth.copy()
#     mesh_depth_viz[~mesh_valid_mask] = np.nan # 原Mesh无效点
    
#     removed_points_viz = np.full_like(lidar_depth, np.nan, dtype=np.float64)
#     removed_points_viz[see_through_condition] = lidar_depth[see_through_condition] # 仅显示被移除的点
    
#     lidar_depth_cleaned_viz = lidar_depth_cleaned.copy().astype(float)
#     lidar_depth_cleaned_viz[lidar_depth_cleaned_viz == 0] = np.nan # 最终所有被设为0的点
    
#     # 确定颜色条的统一最大值
#     vmax = max_depth
#     if np.any(lidar_valid_mask):
#         vmax = np.percentile(lidar_depth[lidar_valid_mask], 98)

#     # 定义绘图数据
#     plot_titles = [
#         '1. Original LiDAR Depth', '2. Mesh Reference Depth',
#         f'3. Removed {num_filtered_points} See-Through Points', '4. Final Cleaned Depth Map (GT)'
#     ]
#     plot_data = [lidar_depth_viz, mesh_depth_viz, removed_points_viz, lidar_depth_cleaned_viz]
#     plot_cmaps = [cmap_viridis, cmap_viridis, cmap_hot, cmap_viridis]

#     # 合并成一个 2x2 的图
#     fig, axes = plt.subplots(2, 2, figsize=(24, 18))
#     fig.suptitle(f"Depth Cleaning Analysis for: {base_name}", fontsize=20)
    
#     for i, ax in enumerate(axes.flat):
#         im = ax.imshow(plot_data[i], cmap=plot_cmaps[i], vmin=0, vmax=vmax)
#         ax.set_title(plot_titles[i], fontsize=16)
#         ax.set_facecolor(INVALID_COLOR)
#         fig.colorbar(im, ax=ax, orientation='vertical', label='Depth (m)', pad=0.02)
        
#     output_path = os.path.join(viz_output_dir, f"{base_name}_visualization.png")
#     plt.savefig(output_path, dpi=150, bbox_inches='tight')
#     plt.close(fig)
    
#     # print(f"    可视化结果已保存到: {output_path}") (在循环中打印太多, 注释掉)


# def main():
#     parser = argparse.ArgumentParser(description="清理激光雷达深度图中的透视点，并将无效值设为0，以便用作GT。")
    
#     # 输入参数
#     parser.add_argument('--lidar_dir', type=str, required=True, help="包含激光雷达深度图的目录。")
#     parser.add_argument('--mesh_dir', type=str, required=True, help="包含Mesh深度图的目录。")
    
#     # 输出参数
#     parser.add_argument('-o', '--output_dir', type=str, required=True, help="保存清理后 .npy 结果的目录。")
    
#     # 算法参数
#     parser.add_argument('-t', '--threshold', type=float, default=3.0, help="判断为透视点的深度差异阈值 (米)。 (默认: 3.0)")
#     parser.add_argument('--max_depth', type=float, default=1000.0, help="最大有效深度 (米)。大于等于此值被视为无效。(默认: 1000.0)")
    
#     # 功能开关
#     parser.add_argument('--no_viz', action='store_true', help="如果设置，则不生成可视化图片，只保存清理后的.npy文件。")

#     args = parser.parse_args()

#     # --- 开始执行 ---
#     print("="*80)
#     print("开始执行深度图清理任务...")
#     print(f"  Lidar 目录: {args.lidar_dir}")
#     print(f"  Mesh 目录:  {args.mesh_dir}")
#     print(f"  输出 (GT) 目录: {args.output_dir}")
#     print(f"  透视阈值: {args.threshold} 米")
#     print(f"  最大有效深度: {args.max_depth} 米")
#     print(f"  生成可视化: {not args.no_viz}")
#     print("="*80)

#     # 创建输出目录
#     os.makedirs(args.output_dir, exist_ok=True)
#     if not args.no_viz:
#         os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)

#     # 查找 Lidar 目录中的所有 .npy 文件
#     lidar_files = sorted(glob.glob(os.path.join(args.lidar_dir, '*.npy')))
#     if not lidar_files:
#         print(f"错误: 在目录 {args.lidar_dir} 中未找到任何 .npy 文件。")
#         sys.exit(1)
        
#     print(f"发现在 Lidar 目录中找到 {len(lidar_files)} 个文件。开始处理...")
    
#     # 遍历每个文件并调用核心处理函数
#     total_files = len(lidar_files)
#     for i, lidar_path in enumerate(lidar_files):
#         base_name = os.path.basename(lidar_path).replace('.npy', '')
#         mesh_path = os.path.join(args.mesh_dir, f"{base_name}.npy")
        
#         # 打印进度
#         print(f"\n--- 进度: ({i+1}/{total_files}) ---")
        
#         if os.path.exists(mesh_path):
#             process_and_visualize_depth(
#                 lidar_path=lidar_path,
#                 mesh_path=mesh_path,
#                 output_dir=args.output_dir,
#                 threshold=args.threshold,
#                 max_depth=args.max_depth,
#                 no_viz=args.no_viz,
#                 base_name=base_name
#             )
#         else:
#             print(f"    警告: 找不到与 {base_name}.npy 匹配的Mesh深度图 (路径: {mesh_path})，跳过。")

#     print("\n" + "="*80)
#     print("所有文件处理完毕！")
#     print(f"清理后的 .npy GT 文件已保存至: {args.output_dir}")
#     if not args.no_viz:
#         print(f"可视化图片已保存至: {os.path.join(args.output_dir, 'visualizations')}")
#     print("="*80)


# if __name__ == '__main__':
#     main()
#!/usr{bin/env python
# -*- coding: utf-8 -*-

# 文件名: clean_depth_parallel.py
# (优化版：支持多核并行、日志文件和进度条)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 文件名: clean_depth_parallel.py (或 szq_5compare-lidar-mesh-depth.py)
# (v2 - 修正了 tqdm TypeError)

import numpy as np
import os
import sys
import glob
import argparse
import subprocess
import logging
import multiprocessing
from joblib import Parallel, delayed

# --- (可选) 自动安装依赖 ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
except ImportError:
    print("未检测到 matplotlib，正在尝试自动安装...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt
        import matplotlib.colors
        print("matplotlib 安装成功。")
    except Exception as e:
        print(f"自动安装失败: {e}\n请手动运行 'pip install matplotlib' 后再试。")
        sys.exit(1)

# (!!!) BUG 修复点 1: 
# (!!!) 从 'import tqdm' 改为 'from tqdm import tqdm'
try:
    from tqdm import tqdm 
except ImportError:
    print("未检测到 tqdm，正在尝试自动安装...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm
    print("tqdm 安装成功。")

try:
    import joblib
except ImportError:
    print("未检测到 joblib，正在尝试自动安装...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib
    print("joblib 安装成功。")


def setup_logging(output_dir):
    """配置日志，分离终端和文件输出"""
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] %(message)s", 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 避免重复添加 handlers (如果在脚本中多次调用)
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.setLevel(logging.INFO) # 设置根级别

    # 1. 日志文件 (INFO 级别)
    log_file_path = os.path.join(output_dir, "processing_log.txt")
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO) # 文件记录所有信息
    root_logger.addHandler(file_handler)

    # 2. 终端 (WARNING 级别)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.WARNING) # 终端只显示重要信息
    root_logger.addHandler(console_handler)

    logging.info("日志系统已启动。")
    logging.info(f"详细日志将写入: {log_file_path}")
    print(f"日志系统已启动。详细日志将写入: {log_file_path}")


def process_and_visualize_depth(lidar_path, mesh_path, output_dir, threshold, max_depth, no_viz, base_name):
    """
    加载、处理并可视化单个激光雷达和Mesh深度图对。
    (此函数现在是并行 "worker"，使用 logging 替代 print)
    """
    
    logging.info(f"--- 正在处理: {base_name}.npy ---")

    try:
        lidar_depth = np.load(lidar_path)
        mesh_depth = np.load(mesh_path)
    except FileNotFoundError as e:
        logging.error(f"文件未找到！ {e}")
        return 
    except Exception as e:
        logging.error(f"加载 {base_name}.npy 时出错: {e}")
        return

    if lidar_depth.shape != mesh_depth.shape:
        logging.warning(f"文件 {base_name} 的尺寸不匹配 ({lidar_depth.shape} vs {mesh_depth.shape})，跳过处理。")
        return

    # --- 2. 核心筛选逻辑 ---
    lidar_valid_mask = (lidar_depth > 0) & (lidar_depth < max_depth)
    mesh_valid_mask = (mesh_depth > 0) & (mesh_depth < max_depth)
    both_valid_mask = lidar_valid_mask & mesh_valid_mask
    
    see_through_condition = np.zeros_like(lidar_depth, dtype=bool)
    if np.any(both_valid_mask):
        see_through_condition[both_valid_mask] = lidar_depth[both_valid_mask] > (mesh_depth[both_valid_mask] + threshold)
    
    num_filtered_points = np.sum(see_through_condition)
    
    # --- 3. 生成并保存清理后的深度图 (关键) ---
    lidar_depth_cleaned = lidar_depth.copy()
    invalid_points_mask = (~lidar_valid_mask) | (~mesh_valid_mask) | see_through_condition
    lidar_depth_cleaned[invalid_points_mask] = 0
    
    output_npy_path = os.path.join(output_dir, f"{base_name}.npy") 
    np.save(output_npy_path, lidar_depth_cleaned)
    
    logging.info(f"处理完成: {base_name} - 找到 {num_filtered_points} 个透视点。")
    logging.info(f"清理后的 GT (npy) 已保存到: {output_npy_path}")

    # --- 4. 可视化 (如果需要) ---
    if no_viz:
        return

    try:
        viz_output_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_output_dir, exist_ok=True)

        INVALID_COLOR = 'darkgray'
        cmap_viridis = plt.get_cmap('viridis').copy()
        cmap_viridis.set_bad(color=INVALID_COLOR)
        cmap_hot = plt.get_cmap('hot').copy()
        cmap_hot.set_bad(color=INVALID_COLOR)

        lidar_depth_viz = lidar_depth.copy()
        lidar_depth_viz[~lidar_valid_mask] = np.nan
        mesh_depth_viz = mesh_depth.copy()
        mesh_depth_viz[~mesh_valid_mask] = np.nan
        removed_points_viz = np.full_like(lidar_depth, np.nan, dtype=np.float64)
        removed_points_viz[see_through_condition] = lidar_depth[see_through_condition]
        lidar_depth_cleaned_viz = lidar_depth_cleaned.copy().astype(float)
        lidar_depth_cleaned_viz[lidar_depth_cleaned_viz == 0] = np.nan
        
        vmax = max_depth
        if np.any(lidar_valid_mask):
            vmax = np.percentile(lidar_depth[lidar_valid_mask], 98)

        plot_titles = [
            '1. Original LiDAR Depth', '2. Mesh Reference Depth',
            f'3. Removed {num_filtered_points} See-Through Points', '4. Final Cleaned Depth Map (GT)'
        ]
        plot_data = [lidar_depth_viz, mesh_depth_viz, removed_points_viz, lidar_depth_cleaned_viz]
        plot_cmaps = [cmap_viridis, cmap_viridis, cmap_hot, cmap_viridis]

        fig, axes = plt.subplots(2, 2, figsize=(24, 18))
        fig.suptitle(f"Depth Cleaning Analysis for: {base_name}", fontsize=20)
        
        for i, ax in enumerate(axes.flat):
            im = ax.imshow(plot_data[i], cmap=plot_cmaps[i], vmin=0, vmax=vmax)
            ax.set_title(plot_titles[i], fontsize=16)
            ax.set_facecolor(INVALID_COLOR)
            fig.colorbar(im, ax=ax, orientation='vertical', label='Depth (m)', pad=0.02)
            
        output_path = os.path.join(viz_output_dir, f"{base_name}_visualization.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logging.info(f"可视化结果已保存到: {output_path}")

    except Exception as e:
        logging.error(f"在为 {base_name} 生成可视化时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="[多核并行版] 清理激光雷达深度图中的透视点。")
    
    parser.add_argument('--lidar_dir', type=str, required=True, help="包含激光雷达深度图的目录。")
    parser.add_argument('--mesh_dir', type=str, required=True, help="包含Mesh深度图的目录。")
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="保存清理后 .npy 结果的目录。")
    
    parser.add_argument('-t', '--threshold', type=float, default=3.0, help="透视点阈值 (米)。(默认: 3.0)")
    parser.add_argument('--max_depth', type=float, default=1000.0, help="最大有效深度 (米)。(默认: 1000.0)")
    
    parser.add_argument('--no_viz', action='store_true', help="如果设置，则不生成可视化图片。")
    
    parser.add_argument('-j', '--num_workers', type=int, default=-1, 
                        help="使用的CPU核心数。-1 表示使用所有核心。(默认: -1)")
    
    args = parser.parse_args()

    # --- 1. 创建目录 ---
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.no_viz:
        os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)

    # --- 2. 设置日志 ---
    setup_logging(args.output_dir)

    # --- 3. 记录启动参数 ---
    logging.info("="*80)
    logging.info("开始执行深度图清理任务 (并行版 v2)")
    logging.info(f"  Lidar 目录: {args.lidar_dir}")
    logging.info(f"  Mesh 目录:  {args.mesh_dir}")
    logging.info(f"  输出 (GT) 目录: {args.output_dir}")
    logging.info(f"  透视阈值: {args.threshold} 米")
    logging.info(f"  最大有效深度: {args.max_depth} 米")
    logging.info(f"  生成可视化: {not args.no_viz}")
    
    num_cores = args.num_workers
    if num_cores == -1:
        num_cores = multiprocessing.cpu_count()
    logging.info(f"  使用核心数 (-j): {num_cores}")
    logging.info("="*80)

    # --- 4. 收集所有任务 ---
    lidar_files = sorted(glob.glob(os.path.join(args.lidar_dir, '*.npy')))
    if not lidar_files:
        logging.error(f"错误: 在目录 {args.lidar_dir} 中未找到任何 .npy 文件。")
        sys.exit(1)
        
    logging.info(f"发现在 Lidar 目录中找到 {len(lidar_files)} 个文件。开始准备任务列表...")

    tasks_to_run = []
    for lidar_path in lidar_files:
        base_name = os.path.basename(lidar_path).replace('.npy', '')
        mesh_path = os.path.join(args.mesh_dir, f"{base_name}.npy")
        
        if os.path.exists(mesh_path):
            tasks_to_run.append((
                lidar_path, 
                mesh_path, 
                args.output_dir, 
                args.threshold, 
                args.max_depth, 
                args.no_viz, 
                base_name
            ))
        else:
            logging.warning(f"警告: 找不到与 {base_name}.npy 匹配的Mesh深度图 (路径: {mesh_path})，跳过。")

    logging.info(f"共 {len(tasks_to_run)} 个有效任务将要执行。")
    if not tasks_to_run:
        logging.warning("没有找到任何匹配的文件对，任务提前结束。")
        print("没有找到任何匹配的文件对，任务提前结束。")
        sys.exit(0)

    # --- 5. 执行并行处理 ---
    
    # (!!!) BUG 修复点 2:
    # (!!!) 删除了错误的、重复的 'with tqdm(...)' 块。
    # (!!!) 只保留这个正确的块。
    
    print(f"开始并行处理 {len(tasks_to_run)} 个文件 (使用 {num_cores} 个核心)...")
    print("正在启动并行池...")
    
    # (!!!) 现在 'tqdm' 可以被直接调用，因为我们使用了 'from tqdm import tqdm'
    Parallel(n_jobs=num_cores)(
        delayed(process_and_visualize_depth)(
            lidar_path=task[0],
            mesh_path=task[1],
            output_dir=task[2],
            threshold=task[3],
            max_depth=task[4],
            no_viz=task[5],
            base_name=task[6]
        ) 
        # TQDM 包装 tasks_to_run
        for task in tqdm(tasks_to_run, desc="清理深度图", unit="file")
    )


    logging.info("\n" + "="*80)
    logging.info("所有文件处理完毕！")
    logging.info(f"清理后的 .npy GT 文件已保存至: {args.output_dir}")
    if not args.no_viz:
        logging.info(f"可视化图片已保存至: {os.path.join(args.output_dir, 'visualizations')}")
    logging.info("="*80)
    
    print("\n所有文件处理完毕！")
    print(f"详细日志请见: {os.path.join(args.output_dir, 'processing_log.txt')}")


if __name__ == '__main__':
    main()
'''


如果 Lidar 原始值无效 (0 或>= max_depth) ➡️ 输出 0

如果 Mesh 原始值无效 (0 或>= max_depth) ➡️ 输出 0

如果 Lidar 和 Mesh 都有效，但Lidar > Mesh + threshold(透视点) ➡️ 输出 0

如果激光雷达和Mesh都有效，且Lidar <= Mesh + threshold(好点)➡️保留激光雷达值

'''
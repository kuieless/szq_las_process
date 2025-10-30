





# # import numpy as np
# # import os
# # import subprocess
# # import sys

# # # --- (可选) 自动安装依赖 ---
# # try:
# #     import matplotlib.pyplot as plt
# # except ImportError:
# #     print("未检测到 matplotlib，正在尝试自动安装...")
# #     try:
# #         subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
# #         import matplotlib.pyplot as plt
# #         print("matplotlib 安装成功。")
# #     except Exception as e:
# #         print(f"自动安装失败: {e}")
# #         print("请手动运行 'pip install matplotlib' 后再试。")
# #         sys.exit(1)

# # # ==============================================================================
# # # --- 1. 参数设置 (请根据需要修改) ---
# # # ==============================================================================

# # # 输入文件路径
# # lidar_depth_path = '/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_dji/000010.npy'  # 您的激光雷达深度图路径
# # mesh_depth_path = '/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_mesh/000010.npy'    # 您的Mesh投影深度图路径

# # # 输出文件路径
# # output_depth_path = 'lidar_depth_cleaned.npy' # 清理后深度图的保存路径
# # output_comparison_image = 'depth_comparison.png' # 对比图的保存路径

# # # 【新】最大有效深度阈值（单位：米）
# # # 所有大于此值的点都将被视为无效
# # MAX_VALID_DEPTH = 1000.0

# # # 透视点判断阈值（单位：米）
# # # 如果 lidar_depth > mesh_depth + SEE_THROUGH_THRESHOLD，则认为是透视点
# # SEE_THROUGH_THRESHOLD = 1.0

# # # ==============================================================================
# # # --- 2. 数据加载 ---
# # # ==============================================================================

# # print("正在加载深度图...")
# # try:
# #     lidar_depth = np.load(lidar_depth_path)
# #     mesh_depth = np.load(mesh_depth_path)
# # except FileNotFoundError as e:
# #     print(f"错误：文件未找到！ {e}")
# #     sys.exit(1)

# # if lidar_depth.shape != mesh_depth.shape:
# #     raise ValueError(f"错误：两个深度图的尺寸不匹配！")

# # print(f"深度图加载成功，尺寸为: {lidar_depth.shape}")

# # # ==============================================================================
# # # --- 3. 核心筛选逻辑 ---
# # # ==============================================================================
# # print("开始执行清理...")

# # # 步骤 1: 定义在每个图中的有效像素（根据新的 MAX_VALID_DEPTH 阈值）
# # lidar_valid_mask = (lidar_depth > 0) & (lidar_depth < MAX_VALID_DEPTH)
# # mesh_valid_mask = (mesh_depth > 0) & (mesh_depth < MAX_VALID_DEPTH)

# # # 步骤 2: 确定可以进行比较的区域 (两个图在同一位置都必须是有效的)
# # both_valid_mask = lidar_valid_mask & mesh_valid_mask

# # # 步骤 3: 在可比较区域内，应用透视点判断条件
# # see_through_condition = lidar_depth[both_valid_mask] > (mesh_depth[both_valid_mask] + SEE_THROUGH_THRESHOLD)

# # # 步骤 4: 创建最终的移除掩码。它只在 'both_valid_mask' 为 True 的地方才可能为 True
# # # 创建一个全为 False 的掩码，然后只在满足条件的位置设为 True
# # final_removal_mask = np.zeros_like(lidar_depth, dtype=bool)
# # final_removal_mask[both_valid_mask] = see_through_condition

# # num_filtered_points = np.sum(final_removal_mask)
# # print(f"处理完成: 找到了 {num_filtered_points} 个透视点并将其标记移除。")

# # # ==============================================================================
# # # --- 4. 生成并保存新的深度图 ---
# # # ==============================================================================

# # # 创建一个副本用于输出
# # lidar_depth_cleaned = lidar_depth.copy()

# # # 步骤 5: 将所有类型的无效点都设置为 0
# # # 5.1 首先，将所有原始的无效点（<=0 或 >=1000）设置为 0
# # lidar_depth_cleaned[~lidar_valid_mask] = 0
# # # 5.2 然后，将被判断为透视点的也设置为 0
# # lidar_depth_cleaned[final_removal_mask] = 0

# # # 保存处理后的 npy 文件
# # np.save(output_depth_path, lidar_depth_cleaned)
# # print(f"清理后的新深度图已保存到: {output_depth_path}")


# # # ==============================================================================
# # # --- 5. 可视化对比 ---
# # # ==============================================================================
# # print("正在生成最终对比图...")

# # # 设置字体以支持中文
# # # plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'sans-serif']
# # # plt.rcParams['axes.unicode_minus'] = False 

# # fig, axes = plt.subplots(4, 1, figsize=(10, 24))
# # fig.suptitle('深度图清理流程对比', fontsize=20, y=1.02)


# # # 为了可视化效果，我们只在有效范围内计算颜色条的最大值
# # valid_original_depth = lidar_depth[lidar_valid_mask]
# # if valid_original_depth.size > 0:
# #     vmax = np.percentile(valid_original_depth, 98) 
# # else:
# #     vmax = MAX_VALID_DEPTH

# # # 子图1: 原始激光雷达深度图
# # im1 = axes[0].imshow(lidar_depth, cmap='viridis', vmax=vmax)
# # axes[0].set_title('1. 原始激光雷达深度图 (无效值可能非常大)')
# # fig.colorbar(im1, ax=axes[0], orientation='vertical', label='Depth (m)')

# # # 子图2: Mesh 参考深度图
# # im2 = axes[1].imshow(mesh_depth, cmap='viridis', vmax=vmax)
# # axes[1].set_title('2. Mesh 参考深度图')
# # fig.colorbar(im2, ax=axes[1], orientation='vertical', label='Depth (m)')

# # # 子图3: 被移除的透视点
# # removed_points_viz = np.zeros_like(lidar_depth)
# # removed_points_viz[final_removal_mask] = lidar_depth[final_removal_mask]
# # im3 = axes[2].imshow(removed_points_viz, cmap='hot', vmax=vmax)
# # axes[2].set_title(f'3. 被识别并移除的 {num_filtered_points} 个透视点')
# # fig.colorbar(im3, ax=axes[2], orientation='vertical', label='Depth (m)')

# # # 子图4: 清理后的激光雷达深度图
# # im4 = axes[3].imshow(lidar_depth_cleaned, cmap='viridis', vmax=vmax)
# # axes[3].set_title('4. 最终清理后的深度图 (无效点和透视点均已设为0)')
# # fig.colorbar(im4, ax=axes[3], orientation='vertical', label='Depth (m)')

# # plt.tight_layout(rect=[0, 0, 1, 0.98])
# # plt.savefig(output_comparison_image, dpi=150, bbox_inches='tight')
# # print(f"最终对比图已保存到: {output_comparison_image}")

# import numpy as np
# import os
# import subprocess
# import sys

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
#         print(f"自动安装失败: {e}")
#         print("请手动运行 'pip install matplotlib' 后再试。")
#         sys.exit(1)

# # ==============================================================================
# # --- 1. 参数设置 (请根据需要修改) ---
# # ==============================================================================

# # 输入文件路径

# # # 输入文件路径
# lidar_depth_path = '/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_dji/000010.npy'  # 您的激光雷达深度图路径
# mesh_depth_path = '/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_mesh/000010.npy'    # 您的Mesh投影深度图路径

# # 输出文件路径
# # 输出文件路径
# output_depth_path = 'lidar_depth_final_cleaned.npy'
# output_comparison_image = 'depth_final_comparison_4in1.png' # 4合1对比图

# # 【新功能】是否保存单张大图
# SAVE_INDIVIDUAL_PLOTS = True

# # 【新功能】用于表示空洞/无效点的醒目颜色
# # 可选值: 'darkgray', 'white', 'black', 'red' 等
# INVALID_COLOR = 'darkgray'

# # 最大有效深度阈值（单位：米）
# MAX_VALID_DEPTH = 1000.0

# # 透视点判断阈值（单位：米）
# SEE_THROUGH_THRESHOLD = 3.0

# # ==============================================================================
# # --- 2. 数据加载与处理 (与上一版相同) ---
# # ==============================================================================

# print("正在加载深度图...")
# # ... (省略与上一版完全相同的数据加载和核心筛选逻辑代码) ...
# try:
#     lidar_depth = np.load(lidar_depth_path)
#     mesh_depth = np.load(mesh_depth_path)
# except FileNotFoundError as e:
#     print(f"错误：文件未找到！ {e}")
#     sys.exit(1)

# if lidar_depth.shape != mesh_depth.shape:
#     raise ValueError(f"错误：两个深度图的尺寸不匹配！")

# print(f"深度图加载成功，尺寸为: {lidar_depth.shape}")
# print("开始执行清理...")
# lidar_valid_mask = (lidar_depth > 0) & (lidar_depth < MAX_VALID_DEPTH)
# mesh_valid_mask = (mesh_depth > 0) & (mesh_depth < MAX_VALID_DEPTH)
# both_valid_mask = lidar_valid_mask & mesh_valid_mask
# see_through_condition = lidar_depth[both_valid_mask] > (mesh_depth[both_valid_mask] + SEE_THROUGH_THRESHOLD)
# final_removal_mask = np.zeros_like(lidar_depth, dtype=bool)
# final_removal_mask[both_valid_mask] = see_through_condition
# num_filtered_points = np.sum(final_removal_mask)
# print(f"处理完成: 找到了 {num_filtered_points} 个透视点并将其标记移除。")
# lidar_depth_cleaned = lidar_depth.copy()
# lidar_depth_cleaned[~lidar_valid_mask] = 0
# lidar_depth_cleaned[final_removal_mask] = 0
# np.save(output_depth_path, lidar_depth_cleaned)
# print(f"清理后的新深度图已保存到: {output_depth_path}")


# # ==============================================================================
# # --- 3. 可视化准备 ---
# # ==============================================================================
# print("正在准备可视化...")

# # --- 创建自定义颜色映射 (Colormap) ---
# # 核心改动：我们创建一个Colormap的副本，并为其设置一个“坏”值颜色。
# # Matplotlib会将所有NaN (Not a Number)值渲染成这个“坏”值颜色。

# # 适用于深度图的 'viridis' Colormap
# cmap_viridis = plt.get_cmap('viridis').copy()
# cmap_viridis.set_bad(color=INVALID_COLOR)

# # 适用于移除点热力图的 'hot' Colormap
# cmap_hot = plt.get_cmap('hot').copy()
# cmap_hot.set_bad(color=INVALID_COLOR)

# # --- 准备用于可视化的数据数组 ---
# # 为了使用set_bad功能，我们需要将所有无效值（0或巨大值）替换为np.nan
# # 我们创建新的 `_viz` 变量，以免修改原始数据

# # 原始激光雷达图：将无效区域变为NaN
# lidar_depth_viz = lidar_depth.copy()
# lidar_depth_viz[~lidar_valid_mask] = np.nan

# # Mesh参考图：将无效区域变为NaN
# mesh_depth_viz = mesh_depth.copy()
# mesh_depth_viz[~mesh_valid_mask] = np.nan

# # 被移除的点：背景(0)变为NaN，只保留被移除的点
# removed_points_viz = np.full_like(lidar_depth, np.nan, dtype=np.float64)
# removed_points_viz[final_removal_mask] = lidar_depth[final_removal_mask]

# # 清理后的图：所有0值（代表无效）都变为NaN
# lidar_depth_cleaned_viz = lidar_depth_cleaned.copy().astype(float)
# lidar_depth_cleaned_viz[lidar_depth_cleaned_viz == 0] = np.nan

# # 计算统一的颜色范围
# valid_original_depth = lidar_depth[lidar_valid_mask]
# vmax = np.percentile(valid_original_depth, 98) if valid_original_depth.size > 0 else MAX_VALID_DEPTH

# # ==============================================================================
# # --- 4. 生成 4合1 对比图 ---
# # ==============================================================================

# # fig, axes = plt.subplots(4, 1, figsize=(10, 24))
# # fig.suptitle('深度图清理流程对比', fontsize=20, y=1.0)

# # plot_titles = [
# #     '1. 原始激光雷达深度图', '2. Mesh 参考深度图',
# #     f'3. 被移除的 {num_filtered_points} 个透视点', '4. 最终清理后的深度图'
# # ]
# # plot_data = [lidar_depth_viz, mesh_depth_viz, removed_points_viz, lidar_depth_cleaned_viz]
# # plot_cmaps = [cmap_viridis, cmap_viridis, cmap_hot, cmap_viridis]

# # for i in range(4):
# #     im = axes[i].imshow(plot_data[i], cmap=plot_cmaps[i], vmax=vmax)
# #     axes[i].set_title(plot_titles[i])
# #     fig.colorbar(im, ax=axes[i], orientation='vertical', label='Depth (m)')
# #     axes[i].set_facecolor(INVALID_COLOR) # 设置坐标轴背景色

# # plt.tight_layout(rect=[0, 0, 1, 0.98])
# # plt.savefig(output_comparison_image, dpi=150)
# # print(f"4合1对比图已保存到: {output_comparison_image}")
# # plt.close(fig) # 关闭图形，释放内存
# # ==============================================================================
# # --- 4. 生成 4合1 对比图 (英文版) ---
# # ==============================================================================

# fig, axes = plt.subplots(4, 1, figsize=(10, 24))
# fig.suptitle('Depth Map Cleaning Workflow', fontsize=20, y=1.0)

# # 使用英文标题
# plot_titles_en = [
#     '1. Original LiDAR Depth',
#     '2. Mesh Reference Depth',
#     f'3. Removed {num_filtered_points} See-Through Points',
#     '4. Final Cleaned Depth Map'
# ]
# plot_data = [lidar_depth_viz, mesh_depth_viz, removed_points_viz, lidar_depth_cleaned_viz]
# plot_cmaps = [cmap_viridis, cmap_viridis, cmap_hot, cmap_viridis]

# for i in range(4):
#     im = axes[i].imshow(plot_data[i], cmap=plot_cmaps[i], vmax=vmax)
#     axes[i].set_title(plot_titles_en[i])
#     fig.colorbar(im, ax=axes[i], orientation='vertical', label='Depth (m)')
#     axes[i].set_facecolor(INVALID_COLOR) # 设置坐标轴背景色

# plt.tight_layout(rect=[0, 0, 1, 0.98])
# plt.savefig(output_comparison_image, dpi=150)
# print(f"4-in-1 comparison image saved to: {output_comparison_image}")
# plt.close(fig) # 关闭图形，释放内存

# # ==============================================================================
# # --- 5. 【新】生成并保存单张大图 (英文版) ---
# # ==============================================================================
# if SAVE_INDIVIDUAL_PLOTS:
#     print("Saving individual large plots...")
#     base_name = os.path.splitext(output_comparison_image)[0].replace('_4in1', '')
    
#     # 遍历四张图的数据和设置
#     for i in range(4):
#         # 创建一个新画布用于绘制单张大图
#         fig_ind, ax_ind = plt.subplots(figsize=(12, 9)) # 使用更适合单图的比例
        
#         im = ax_ind.imshow(plot_data[i], cmap=plot_cmaps[i], vmax=vmax)
#         ax_ind.set_title(plot_titles_en[i], fontsize=16)
#         ax_ind.set_facecolor(INVALID_COLOR)
        
#         # 添加颜色条
#         cbar = fig_ind.colorbar(im, orientation='vertical', label='Depth (m)', pad=0.02)
#         cbar.ax.tick_params(labelsize=10)
        
#         # 定义文件名并保存
#         # 生成更简洁的文件名, e.g., "plot_1_Original.png"
#         simple_title = plot_titles_en[i].split(' ')[1] 
#         output_path = f"{base_name}_{i+1}_{simple_title}.png"
        
#         plt.savefig(output_path, dpi=200, bbox_inches='tight')
#         plt.close(fig_ind) # 关闭图形，释放内存
#         print(f" -> Saved: {output_path}")

# print("All tasks completed!")

# # ==============================================================================
# # --- 5. 【新】生成并保存单张大图 ---
# # ==============================================================================
# if SAVE_INDIVIDUAL_PLOTS:
#     print("正在保存单张大图...")
#     base_name = os.path.splitext(output_comparison_image)[0]
    
#     # 遍历四张图的数据和设置
#     for i in range(4):
#         # 创建一个新画布用于绘制单张大图
#         fig_ind, ax_ind = plt.subplots(figsize=(12, 9)) # 使用更适合单图的比例
        
#         im = ax_ind.imshow(plot_data[i], cmap=plot_cmaps[i], vmax=vmax)
#         ax_ind.set_title(plot_titles[i], fontsize=16)
#         ax_ind.set_facecolor(INVALID_COLOR)
        
#         # 添加颜色条
#         cbar = fig_ind.colorbar(im, orientation='vertical', label='Depth (m)', pad=0.02)
#         cbar.ax.tick_params(labelsize=10)
        
#         # 定义文件名并保存
#         output_path = f"{base_name}_{i+1}_{plot_titles[i].split(' ')[0]}.png"
#         plt.savefig(output_path, dpi=200, bbox_inches='tight')
#         plt.close(fig_ind) # 关闭图形，释放内存
#         print(f" -> 已保存: {output_path}")

# print("所有任务完成！")

import numpy as np
import os
import sys
import glob
import argparse
import subprocess

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

def process_and_visualize_depth(lidar_path, mesh_path, output_dir, threshold, max_depth, no_viz):
    """
    加载、处理并可视化单个激光雷达和Mesh深度图对。
    """
    base_name = os.path.basename(lidar_path).replace('.npy', '')
    print(f"\n--- Processing: {base_name} ---")

    # --- 1. 数据加载 ---
    try:
        lidar_depth = np.load(lidar_path)
        mesh_depth = np.load(mesh_path)
    except FileNotFoundError as e:
        print(f"错误：文件未找到！ {e}")
        return

    if lidar_depth.shape != mesh_depth.shape:
        print(f"错误：文件 {base_name} 的尺寸不匹配，跳过处理。")
        return

    # --- 2. 核心筛选逻辑 ---
    lidar_valid_mask = (lidar_depth > 0) & (lidar_depth < max_depth)
    mesh_valid_mask = (mesh_depth > 0) & (mesh_depth < max_depth)
    both_valid_mask = lidar_valid_mask & mesh_valid_mask
    see_through_condition = lidar_depth[both_valid_mask] > (mesh_depth[both_valid_mask] + threshold)
    
    final_removal_mask = np.zeros_like(lidar_depth, dtype=bool)
    final_removal_mask[both_valid_mask] = see_through_condition
    num_filtered_points = np.sum(final_removal_mask)
    
    # --- 3. 生成并保存清理后的深度图 ---
    lidar_depth_cleaned = lidar_depth.copy()
    lidar_depth_cleaned[~lidar_valid_mask] = 0
    lidar_depth_cleaned[final_removal_mask] = 0
    
    output_npy_path = os.path.join(output_dir, f"{base_name}_cleaned.npy")
    np.save(output_npy_path, lidar_depth_cleaned)
    print(f"处理完成: 找到 {num_filtered_points} 个透视点。")
    print(f"清理后的深度图已保存到: {output_npy_path}")

    # --- 4. 可视化 (如果需要) ---
    if no_viz:
        return

    # 可视化准备
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
    removed_points_viz[final_removal_mask] = lidar_depth[final_removal_mask]
    lidar_depth_cleaned_viz = lidar_depth_cleaned.copy().astype(float)
    lidar_depth_cleaned_viz[lidar_depth_cleaned_viz == 0] = np.nan

    valid_original_depth = lidar_depth[lidar_valid_mask]
    vmax = np.percentile(valid_original_depth, 98) if valid_original_depth.size > 0 else max_depth

    # 定义绘图数据
    plot_titles = [
        '1. Original LiDAR Depth', '2. Mesh Reference Depth',
        f'3. Removed {num_filtered_points} See-Through Points', '4. Final Cleaned Depth Map'
    ]
    plot_data = [lidar_depth_viz, mesh_depth_viz, removed_points_viz, lidar_depth_cleaned_viz]
    plot_cmaps = [cmap_viridis, cmap_viridis, cmap_hot, cmap_viridis]

    # 生成并保存单张大图
    for i in range(4):
        fig_ind, ax_ind = plt.subplots(figsize=(12, 9))
        im = ax_ind.imshow(plot_data[i], cmap=plot_cmaps[i], vmax=vmax)
        ax_ind.set_title(plot_titles[i], fontsize=16)
        ax_ind.set_facecolor(INVALID_COLOR)
        cbar = fig_ind.colorbar(im, orientation='vertical', label='Depth (m)', pad=0.02)
        
        simple_title = plot_titles[i].split(' ')[1]
        output_path = os.path.join(output_dir, f"{base_name}_{i+1}_{simple_title}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig_ind)
    
    print(f"可视化结果已保存到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="清理激光雷达深度图中的透视点，并生成可视化结果。")
    
    # 输入参数 (文件或目录)
    parser.add_argument('-l', '--lidar', type=str, help="单个激光雷达深度图文件路径 (.npy)。")
    parser.add_argument('-m', '--mesh', type=str, help="单个Mesh深度图文件路径 (.npy)。")
    parser.add_argument('--lidar_dir', type=str, help="包含激光雷达深度图的目录。")
    parser.add_argument('--mesh_dir', type=str, help="包含Mesh深度图的目录。")
    
    # 输出参数
    parser.add_argument('-o', '--output_dir', type=str, default="./results", help="保存结果的目录。")
    
    # 算法参数
    parser.add_argument('-t', '--threshold', type=float, default=3.0, help="判断为透视点的深度差异阈值 (米)。")
    parser.add_argument('--max_depth', type=float, default=1000.0, help="最大有效深度 (米)。")
    
    # 功能开关
    parser.add_argument('--no_viz', action='store_true', help="如果设置，则不生成可视化图片，只保存清理后的.npy文件。")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 判断是处理单个文件还是整个目录
    if args.lidar and args.mesh:
        process_and_visualize_depth(args.lidar, args.mesh, args.output_dir, args.threshold, args.max_depth, args.no_viz)
    elif args.lidar_dir and args.mesh_dir:
        lidar_files = sorted(glob.glob(os.path.join(args.lidar_dir, '*.npy')))
        if not lidar_files:
            print(f"错误: 在目录 {args.lidar_dir} 中未找到任何 .npy 文件。")
            return
            
        for lidar_path in lidar_files:
            base_name = os.path.basename(lidar_path)
            mesh_path = os.path.join(args.mesh_dir, base_name)
            if os.path.exists(mesh_path):
                process_and_visualize_depth(lidar_path, mesh_path, args.output_dir, args.threshold, args.max_depth, args.no_viz)
            else:
                print(f"警告: 找不到与 {base_name} 匹配的Mesh深度图，跳过。")
    else:
        print("错误: 请提供一对输入文件 (--lidar 和 --mesh) 或一对输入目录 (--lidar_dir 和 --mesh_dir)。")
        parser.print_help()

if __name__ == '__main__':
    main()

'''
python compare-lidar-mesh-depth.py \
    -l /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_dji/000010.npy \
    -m /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_mesh/000010.npy \
    -o ./my_single_file_results \
    -t 3.0

python compare-lidar-mesh-depth.py \
    --lidar_dir /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_dji \
    --mesh_dir /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_mesh \
    -o ./my_batch_results \
    -t 3.0

'''
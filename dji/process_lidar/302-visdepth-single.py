# import numpy as np
# import cv2
# import argparse
# from pathlib import Path

# def create_depth_colorbar(height, width, min_val, max_val, colormap=cv2.COLORMAP_JET):
#     """创建一个带有刻度和米制单位的颜色标尺图像。"""
#     # 创建一个从上到下的颜色渐变条
#     gradient = np.arange(height, 0, -1, dtype=np.float32).reshape(height, 1)
#     gradient = (gradient / height * 255).astype(np.uint8)
#     color_bar_img = cv2.applyColorMap(gradient, colormap)
#     color_bar_img = cv2.resize(color_bar_img, (width, height))

#     # 创建一个用于绘制文本的白色背景
#     text_canvas = np.full((height, 120, 3), 255, dtype=np.uint8)
    
#     # 定义文本样式
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_color = (0, 0, 0) # 黑色
    
#     # 绘制刻度和标签
#     num_labels = 7
#     for i in range(num_labels):
#         p = i / (num_labels - 1)  # 从上到下的百分比 (0.0 to 1.0)
#         y = int(p * (height - 20)) + 10  # y坐标
#         val = max_val * (1 - p) + min_val * p  # 对应的深度值
        
#         # 绘制文本标签
#         cv2.putText(text_canvas, f"{val:.2f} m", (10, y + 5), font, 0.6, font_color, 2, cv2.LINE_AA)
#         # 绘制刻度线
#         cv2.line(color_bar_img, (width-15, y), (width, y), (255, 255, 255), 2)
            
#     return np.hstack([color_bar_img, text_canvas])

# def main(args):
#     input_path = Path(args.input_path)
    
#     # --- 1. 检查和加载文件 ---
#     if not input_path.is_file() or input_path.suffix != '.npy':
#         print(f"❌ 错误: 输入路径不是一个有效的 .npy 文件 -> {input_path}")
#         return

#     print(f"📄 正在加载: {input_path}")
#     depth_map = np.load(input_path)
    
#     # 移除单维度，例如 (H, W, 1) -> (H, W)
#     depth_map = np.squeeze(depth_map)
    
#     if depth_map.ndim != 2:
#         print(f"❌ 错误: .npy 文件中的数据不是一个二维数组 (深度图)，形状为 {depth_map.shape}")
#         return

#     # --- 2. 分析深度数据 ---
#     # 假设有效深度值大于0
#     valid_mask = depth_map > 0
#     valid_depths = depth_map[valid_mask]

#     if valid_depths.size == 0:
#         print("⚠️ 警告: 深度图中未找到任何有效的正数深度值。")
#         min_depth, max_depth = 0.0, 1.0 # 使用默认范围
#     else:
#         # 使用百分位数来确定颜色范围，这能很好地抵抗极端异常值
#         min_depth = np.percentile(valid_depths, 2)
#         max_depth = np.percentile(valid_depths, 98)
#         # 防止min和max相等
#         if min_depth >= max_depth:
#             max_depth = min_depth + 1.0

#     print(f"📊 自动检测到的深度范围 (2%-98%): {min_depth:.2f}m - {max_depth:.2f}m")

#     # --- 3. 创建可视化图像 ---
#     # 创建一个用于可视化的深度图副本
#     depth_vis = depth_map.copy()
    
#     # 将数值裁剪并归一化到 0-255
#     depth_vis[depth_vis < min_depth] = min_depth
#     depth_vis[depth_vis > max_depth] = max_depth
    
#     # 归一化到 0-1
#     depth_vis = (depth_vis - min_depth) / (max_depth - min_depth)
#     depth_vis = (depth_vis * 255).astype(np.uint8)
    
#     # 应用伪彩色映射
#     colorized_depth = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    
#     # 将所有无效区域（例如天空、0值）设为黑色
#     colorized_depth[~valid_mask] = [0, 0, 0]

#     # --- 4. 创建颜色标尺并合并图像 ---
#     h, w, _ = colorized_depth.shape
#     color_bar = create_depth_colorbar(height=h, width=80, min_val=min_depth, max_val=max_depth)
    
#     combined_image = np.hstack([colorized_depth, color_bar])

#     # --- 5. 保存结果 ---
#     if args.output_path:
#         output_path = Path(args.output_path)
#     else:
#         # 如果未指定输出路径，则在输入文件同目录下生成
#         output_path = input_path.parent / f"{input_path.stem}_visualization.png"
    
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(output_path), combined_image)
#     print(f"✅ 可视化图像已保存到: {output_path}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="将单个NPY深度图文件可视化为带颜色标尺的彩色图像。")
#     parser.add_argument("input_path", type=str, help="输入的 .npy 深度图文件路径。")
#     parser.add_argument("-o", "--output_path", type=str, help="(可选) 输出的 .png 图像文件路径。")
    
#     args = parser.parse_args()
#     main(args)
# '''
# python 302-visdepth-single.py "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sztu/depth_gt_sztu" -o ./my_results

# python 302-visdepth-single.py /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/my_single_file_results3/000010_cleaned.npy
# '''



import numpy as np
import cv2
import argparse
from pathlib import Path
import sys

def create_depth_colorbar(height, width, min_val, max_val, colormap=cv2.COLORMAP_JET):
    """创建一个带有刻度和米制单位的颜色标尺图像。"""
    # 创建一个从上到下的颜色渐变条
    gradient = np.arange(height, 0, -1, dtype=np.float32).reshape(height, 1)
    gradient = (gradient / height * 255).astype(np.uint8)
    color_bar_img = cv2.applyColorMap(gradient, colormap)
    color_bar_img = cv2.resize(color_bar_img, (width, height))

    # 创建一个用于绘制文本的白色背景
    text_canvas = np.full((height, 120, 3), 255, dtype=np.uint8)
    
    # 定义文本样式
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 0) # 黑色
    
    # 绘制刻度和标签
    num_labels = 7
    for i in range(num_labels):
        p = i / (num_labels - 1)  # 从上到下的百分比 (0.0 to 1.0)
        y = int(p * (height - 20)) + 10  # y坐标
        val = max_val * (1 - p) + min_val * p  # 对应的深度值
        
        # 绘制文本标签
        cv2.putText(text_canvas, f"{val:.2f} m", (10, y + 5), font, 0.6, font_color, 2, cv2.LINE_AA)
        # 绘制刻度线
        cv2.line(color_bar_img, (width-15, y), (width, y), (255, 255, 255), 2)
            
    return np.hstack([color_bar_img, text_canvas])

def process_depth_file(input_file_path: Path, output_file_path: Path):
    """
    加载、处理并可视化单个NPY深度图文件。
    """
    try:
        # --- 1. 检查和加载文件 ---
        # 路径有效性检查已在 main 中完成
        print(f"📄 正在加载: {input_file_path}")
        depth_map = np.load(input_file_path)
        
        # 移除单维度，例如 (H, W, 1) -> (H, W)
        depth_map = np.squeeze(depth_map)
        
        if depth_map.ndim != 2:
            print(f"❌ 错误: {input_file_path.name} 中的数据不是一个二维数组 (深度图)，形状为 {depth_map.shape}")
            return False

        # --- 2. 分析深度数据 ---
        # 假设有效深度值大于0
        valid_mask = depth_map > 0
        valid_depths = depth_map[valid_mask]

        if valid_depths.size == 0:
            print(f"⚠️ 警告: {input_file_path.name} 中未找到任何有效的正数深度值。")
            min_depth, max_depth = 0.0, 1.0 # 使用默认范围
        else:
            # 使用百分位数来确定颜色范围，这能很好地抵抗极端异常值
            min_depth = np.percentile(valid_depths, 2)
            max_depth = np.percentile(valid_depths, 98)
            # 防止min和max相等
            if min_depth >= max_depth:
                max_depth = min_depth + 1.0

        print(f"📊 自动检测到的深度范围 (2%-98%): {min_depth:.2f}m - {max_depth:.2f}m")

        # --- 3. 创建可视化图像 ---
        # 创建一个用于可视化的深度图副本
        depth_vis = depth_map.copy()
        
        # 将数值裁剪并归一化到 0-255
        depth_vis[depth_vis < min_depth] = min_depth
        depth_vis[depth_vis > max_depth] = max_depth
        
        # 归一化到 0-1
        depth_vis = (depth_vis - min_depth) / (max_depth - min_depth)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        
        # 应用伪彩色映射
        colorized_depth = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        # 将所有无效区域（例如天空、0值）设为黑色
        colorized_depth[~valid_mask] = [0, 0, 0]

        # --- 4. 创建颜色标尺并合并图像 ---
        h, w, _ = colorized_depth.shape
        color_bar = create_depth_colorbar(height=h, width=80, min_val=min_depth, max_val=max_depth)
        
        combined_image = np.hstack([colorized_depth, color_bar])

        # --- 5. 保存结果 ---
        # output_file_path 已由 main 函数确定
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file_path), combined_image)
        print(f"✅ 可视化图像已保存到: {output_file_path}")
        return True

    except Exception as e:
        print(f"❌ 处理 {input_file_path.name} 时发生意外错误: {e}")
        return False

def main(args):
    input_path = Path(args.input_path)
    output_arg = Path(args.output_path) if args.output_path else None

    # --- 1. 检查输入路径是文件还是目录 ---
    
    if not input_path.exists():
        print(f"❌ 错误: 输入路径不存在 -> {input_path}")
        sys.exit(1)

    # --- 情况 A: 输入是单个文件 ---
    if input_path.is_file():
        if input_path.suffix != '.npy':
            print(f"❌ 错误: 输入文件不是一个 .npy 文件 -> {input_path}")
            sys.exit(1)
        
        # 确定最终的输出文件路径
        if output_arg:
            if output_arg.suffix == "": # -o 指定的是一个目录
                output_arg.mkdir(parents=True, exist_ok=True)
                final_output_file = output_arg / f"{input_path.stem}_visualization.png"
            else: # -o 指定的是一个完整的文件路径
                final_output_file = output_arg
        else: # 未指定 -o，保存在输入文件旁边
            final_output_file = input_path.parent / f"{input_path.stem}_visualization.png"
        
        print(f"--- 模式: 单文件处理 ---")
        process_depth_file(input_path, final_output_file)

    # --- 情况 B: 输入是一个目录 ---
    elif input_path.is_dir():
        # 检查输出路径（如果提供了）是否也是一个目录
        if output_arg and output_arg.suffix != "":
            print(f"❌ 错误: 输入是一个目录，但输出 (-o) 被指定为一个文件。")
            print("请指定一个输出目录或不指定 -o (将使用默认输出目录)。")
            sys.exit(1)

        # 确定输出目录
        output_dir = output_arg or input_path.parent / f"{input_path.name}_visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"--- 模式: 批量处理目录 ---")
        print(f"📁 正在扫描目录: {input_path}")
        print(f"💾 结果将保存到: {output_dir}")

        npy_files = sorted(list(input_path.glob('*.npy')))
        
        if not npy_files:
            print(f"⚠️ 警告: 在 {input_path} 中未找到任何 .npy 文件。")
            return

        print(f"🔍 找到 {len(npy_files)} 个 .npy 文件。开始批量处理...")
        
        success_count = 0
        fail_count = 0
        
        for i, npy_file in enumerate(npy_files):
            print(f"\n--- [ {i+1} / {len(npy_files)} ] ---")
            final_output_file = output_dir / f"{npy_file.stem}_visualization.png"
            if process_depth_file(npy_file, final_output_file):
                success_count += 1
            else:
                fail_count += 1
        
        print("\n--- 批量处理完成 ---")
        print(f"✅ 成功: {success_count}")
        print(f"❌ 失败: {fail_count}")

    # --- 情况 C: 路径无效 ---
    else:
        print(f"❌ 错误: 输入路径既不是文件也不是目录 -> {input_path}")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将NPY深度图（或目录）可视化为带颜色标尺的彩色图像。")
    parser.add_argument("input_path", type=str, 
                        help="输入的 .npy 深度图文件路径，或包含 .npy 文件的目录路径。")
    parser.add_argument("-o", "--output_path", type=str, 
                        help="(可选) 输出路径。如果输入是文件，这可以是目标 .png 文件或目标目录。"
                             "如果输入是目录，这必须是目标目录。")
    
    args = parser.parse_args()
    main(args)


    '''

# 示例1：处理单个文件（与之前相同）
python 302-visdepth-single.py /path/to/my/000010_cleaned.npy -o ./my_single_result.png

# 示例2：处理单个文件，并让它自动命名（通过指定输出目录）
python 302-visdepth-single.py /path/to/my/000010_cleaned.npy -o ./my_results/

# 示例3：处理整个目录（默认输出）
# (这将在 /path/to/my_depths 的同级目录创建一个 "my_depths_visualizations" 目录)
python 302-visdepth-single.py /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/SMBU/depth_gt_SMBU

# 示例4：处理整个目录（指定输出目录）
python 302-visdepth-single.py /path/to/my_depths -o ./all_my_visualizations


    '''
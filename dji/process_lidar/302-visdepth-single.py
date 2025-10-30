import numpy as np
import cv2
import argparse
from pathlib import Path

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

def main(args):
    input_path = Path(args.input_path)
    
    # --- 1. 检查和加载文件 ---
    if not input_path.is_file() or input_path.suffix != '.npy':
        print(f"❌ 错误: 输入路径不是一个有效的 .npy 文件 -> {input_path}")
        return

    print(f"📄 正在加载: {input_path}")
    depth_map = np.load(input_path)
    
    # 移除单维度，例如 (H, W, 1) -> (H, W)
    depth_map = np.squeeze(depth_map)
    
    if depth_map.ndim != 2:
        print(f"❌ 错误: .npy 文件中的数据不是一个二维数组 (深度图)，形状为 {depth_map.shape}")
        return

    # --- 2. 分析深度数据 ---
    # 假设有效深度值大于0
    valid_mask = depth_map > 0
    valid_depths = depth_map[valid_mask]

    if valid_depths.size == 0:
        print("⚠️ 警告: 深度图中未找到任何有效的正数深度值。")
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
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        # 如果未指定输出路径，则在输入文件同目录下生成
        output_path = input_path.parent / f"{input_path.stem}_visualization.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined_image)
    print(f"✅ 可视化图像已保存到: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将单个NPY深度图文件可视化为带颜色标尺的彩色图像。")
    parser.add_argument("input_path", type=str, help="输入的 .npy 深度图文件路径。")
    parser.add_argument("-o", "--output_path", type=str, help="(可选) 输出的 .png 图像文件路径。")
    
    args = parser.parse_args()
    main(args)
'''
python 302-visdepth-single.py "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_dji/000010.npy" -o ./my_results
python 302-visdepth-single.py /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/my_single_file_results3/000010_cleaned.npy
'''
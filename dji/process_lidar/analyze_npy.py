#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import sys
import os

def analyze_npy(filepath, custom_range=None, outlier_range=None):
    """
    加载并分析 .npy 文件，提供基本信息、零值、NaN/Inf、自定义范围和奇值（范围外）的统计。
    """
    
    # --- 1. 加载数据 ---
    if not os.path.exists(filepath):
        print(f"错误: 文件未找到 {filepath}", file=sys.stderr)
        sys.exit(1)
        
    try:
        data = np.load(filepath)
    except Exception as e:
        print(f"错误: 加载文件时出错 {filepath}. 错误信息: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 2. 基本信息 ---
    print(f"--- 正在分析: {filepath} ---")
    
    # 检查是否为非数值类型
    if not np.issubdtype(data.dtype, np.number):
        print("\n--- 基本信息 ---")
        print(f"数据类型 (dtype): {data.dtype}")
        print(f"形状 (shape): {data.shape}")
        print(f"元素总数: {data.size:,}")
        print("\n错误: 数组为非数值类型，无法执行统计分析。")
        sys.exit(1)

    # 展平以便于统计（如果需要处理多维数组的统计）
    # data_flat = data.ravel()
    total_elements = data.size
    
    # 防止在空数组或全为NaN的数组上出错
    try:
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        mean_val = np.nanmean(data)
        std_val = np.nanstd(data)
        median_val = np.nanmedian(data)
    except ValueError as e:
        # 可能是空数组
        print(f"警告: 计算基本统计时出错 (可能是空数组): {e}")
        min_val = max_val = mean_val = std_val = median_val = np.nan

    print("\n--- 1. 基本信息 ---")
    print(f"数据类型 (dtype): {data.dtype}")
    print(f"形状 (shape): {data.shape}")
    print(f"元素总数: {total_elements:,}")
    print(f"最小值 (Min): {min_val:,.4f}")
    print(f"最大值 (Max): {max_val:,.4f}")
    print(f"平均值 (Mean): {mean_val:,.4f}")
    print(f"中位数 (Median): {median_val:,.4f}")
    print(f"标准差 (Std Dev): {std_val:,.4f}")

    if total_elements == 0:
        print("\n数组为空，无法进行进一步分析。")
        return

    # --- 3. 奇值和零值分析 ---
    print("\n--- 2. 奇值 / 特殊值分析 ---")

    # 零值分析
    zero_count = np.sum(data == 0)
    zero_percent = (zero_count / total_elements) * 100
    print(f"零值 (== 0): {zero_count:,} 个, 占比: {zero_percent:.4f}%")
    
    # 近似零值 (例如 < 1e-9)
    # close_zero_count = np.sum(np.isclose(data, 0, atol=1e-9))
    # close_zero_percent = (close_zero_count / total_elements) * 100
    # print(f"近似零值 (< 1e-9): {close_zero_count:,} 个, 占比: {close_zero_percent:.4f}%")

    # NaN (Not a Number)
    nan_count = np.sum(np.isnan(data))
    nan_percent = (nan_count / total_elements) * 100
    print(f"NaN (非数值): {nan_count:,} 个, 占比: {nan_percent:.4f}%")

    # Inf (无穷大)
    inf_count = np.sum(np.isinf(data))
    inf_percent = (inf_count / total_elements) * 100
    print(f"Inf (无穷大): {inf_count:,} 个, 占比: {inf_percent:.4f}%")

    # --- 4. 自定义范围分析 ---
    print("\n--- 3. 自定义范围分析 ---")
    
    if custom_range:
        r_min, r_max = sorted(custom_range) # 确保min <= max
        # 使用逻辑与 (logical_and) 或 &
        range_count = np.sum((data >= r_min) & (data <= r_max))
        range_percent = (range_count / total_elements) * 100
        print(f"范围 [{r_min}, {r_max}] 内: {range_count:,} 个, 占比: {range_percent:.4f}%")
    else:
        print("未指定 --range, 跳过范围分析。")

    # --- 5. 奇值点（范围外）分析 ---
    print("\n--- 4. 奇值点 (范围外) 分析 ---")
    
    if outlier_range:
        o_min, o_max = sorted(outlier_range) # 确保min <= max
        # 使用逻辑或 (logical_or) 或 |
        # 统计所有 *在* [o_min, o_max] 范围 *之外* 的点
        outlier_count = np.sum((data < o_min) | (data > o_max))
        outlier_percent = (outlier_count / total_elements) * 100
        print(f"范围 [{o_min}, {o_max}] 外: {outlier_count:,} 个, 占比: {outlier_percent:.4f}%")
    else:
        print("未指定 --outlier_range, 跳过奇值点分析。")
        print("提示: 可使用 --outlier_range 设定一个 '正常' 范围 (例如 -1000 1000)，")
        print("      脚本将统计所有在该范围之外的 '非常大' 或 '非常小' 的值。")

    print("\n--- 分析完成 ---")


if __name__ == "__main__":
    # --- Argparse 设置 ---
    parser = argparse.ArgumentParser(
        description="分析 .npy 文件的基本信息、零值、自定义范围和奇值。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "filepath", 
        type=str, 
        help="要分析的 .npy 文件的路径。"
    )
    
    parser.add_argument(
        "-r", "--range", 
        nargs=2, 
        type=float, 
        metavar=('MIN', 'MAX'),
        help="指定一个你关心的范围 [MIN, MAX]，脚本将统计落在这个范围内的元素百分比。\n"
             "例如: -r 0.1 1.0"
    )
    
    parser.add_argument(
        "-o", "--outlier_range", 
        nargs=2, 
        type=float, 
        metavar=('LOW', 'HIGH'),
        help="指定一个 '正常' 值的范围 [LOW, HIGH]。\n"
             "脚本将统计所有 *小于* LOW 或 *大于* HIGH 的奇值点 (Outlier) 的百分比。\n"
             "例如: -o -1000 1000"
    )

    args = parser.parse_args()
    
    analyze_npy(args.filepath, args.range, args.outlier_range)

    #python analyze_npy.py /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/hav/depth_metric/000000.npy 
    # python analyze_npy.py  /home/data1/szq/Megadepth/UAV_downsampled_504/ainterval5_AMtown01_cropped_downsampled_downsampled/#python analyze_npy.py /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/hav/depth_mesh/000005.npy --outlier_range 0 1000
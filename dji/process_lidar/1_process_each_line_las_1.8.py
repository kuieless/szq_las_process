# 脚本功能：
# 1. 加载并按EXIF时间排序照片元数据。
# 2. (核心修改) 提供非对称的点云划分策略：
#    - 用户可以自定义照片时间戳 (T) 之前的秒数 (n) 和之后的秒数 (m)。
#    - 划分窗口为 [T - n, T + m)。
# 3. 用户可以通过修改顶部的配置变量 (SECONDS_BEFORE, SECONDS_AFTER) 轻松配置。
# 4. 输出分段后的点云列表 (points_lidar_list.pkl)。

from plyfile import PlyData
import numpy as np
from tqdm import tqdm

import laspy
import torch
import datetime
from pathlib import Path
import pickle
import os
import sys

# --- 项目路径设置 (与您的脚本保持一致) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)
from dji.opt import _get_opts


# --- 辅助函数，用于解析曝光时间字符串 ---
def parse_exposure_time(value):
    """(已修正) 将 ['1/2000'] 或 '1/2000' 这样的值转换为秒(float)"""
    # 检查输入是否为列表，如果是，则提取第一个元素
    if isinstance(value, list) and len(value) > 0:
        time_str = value[0]
    elif isinstance(value, str):
        time_str = value
    else:
        # 如果输入既不是列表也不是字符串，无法处理
        print(f"警告: 曝光时间格式未知 '{value}', 将使用默认值 0.001 秒。")
        return 0.001

    try:
        if '/' in time_str:
            num, den = map(float, time_str.split('/'))
            return num / den
        else:
            return float(time_str)
    except (ValueError, ZeroDivisionError, TypeError):
        # 增加了对TypeError的捕获，以防万一
        print(f"警告: 无法解析曝光时间字符串 '{time_str}', 将使用默认值 0.001 秒。")
        return 0.001


def main(hparams):
    # ==============================================================================
    # --- 新增配置：在这里设置您的非对称时间窗口 ---
    #
    # 示例:
    # 假设照片时间戳为 T
    # SECONDS_BEFORE = 2.0  (n=2.0) -> 窗口开始时间 = T - 2.0 秒
    # SECONDS_AFTER  = 1.0  (m=1.0) -> 窗口结束时间 = T + 1.0 秒
    # 总窗口大小 = 3.0 秒

    SECONDS_BEFORE = 0.0
    SECONDS_AFTER = 7.0
    # ==============================================================================

    dataset_path = Path(hparams.dataset_path)
    las_output_path = Path(hparams.las_output_path)
    las_output_path.mkdir(parents=True, exist_ok=True)

    # --- 1. 加载LiDAR点云数据 ---
    print("正在加载LAS文件...")
    in_file = laspy.file.File(hparams.las_path, mode='r')

    x, y, z = in_file.x, in_file.y, in_file.z
    r = (in_file.red / 65536.0 * 255).astype(np.uint8)
    g = (in_file.green / 65536.0 * 255).astype(np.uint8)
    b = (in_file.blue / 65536.0 * 255).astype(np.uint8)
    lidars = np.array(list(zip(x, y, z, r, g, b)))

    diff_gps_time = np.array(in_file.gps_time - in_file.gps_time[0])
    print("LAS文件加载并计算相对时间戳完成。")

    # --- 2. 加载并按EXIF时间排序照片元数据 ---
    train_path_candidates = sorted(list((dataset_path / 'train' / 'image_metadata').iterdir()))
    val_path_candidates = sorted(list((dataset_path / 'val' / 'image_metadata').iterdir()))
    all_metadata_paths = train_path_candidates + val_path_candidates

    print("\n正在加载所有图像元数据并按EXIF拍摄时间排序...")
    all_metadata = []
    for path in tqdm(all_metadata_paths, desc="Loading metadata"):
        metadata = torch.load(path, weights_only=False)
        all_metadata.append({'data': metadata, 'path': path})

    all_metadata.sort(key=lambda x: x['data']['meta_tags']['EXIF DateTimeOriginal'].values)
    print("照片元数据排序完成。")

    # --- 3. 提取精确时间戳 ---
    # (注意: 曝光时间 exposure_time 在这个版本中不再用于划分，但我们保留提取逻辑以备将来使用)
    print("\n正在提取每张照片的精确时间...")
    sorted_photo_info = []
    origin_timestamp = None

    for item in tqdm(all_metadata, desc="Extracting photo info"):
        metadata = item['data']
        time_str = metadata['meta_tags']['EXIF DateTimeOriginal'].values
        dt_local = datetime.datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
        unix_timestamp = dt_local.timestamp()

        if origin_timestamp is None:
            origin_timestamp = unix_timestamp
        time_difference = unix_timestamp - origin_timestamp

        # (我们仍然解析曝光时间，以防未来需要，但它不用于划分)
        exposure_time_key = 'EXIF ExposureTime'
        if exposure_time_key in metadata['meta_tags']:
            exposure_time_value = metadata['meta_tags'][exposure_time_key].values
            exposure_time_sec = parse_exposure_time(exposure_time_value)
        else:
            exposure_time_sec = 0.001

        sorted_photo_info.append({
            "relative_time": time_difference,
            "exposure_time": exposure_time_sec
        })

    # --- 4. (核心修改) 根据选择的非对称策略进行点云划分 ---
    print("\n" + "=" * 25 + " 点云划分配置 " + "=" * 25)
    print(f"划分策略: 不对称时间窗口")
    print(f"照片时间戳 (T) 之前: {SECONDS_BEFORE} 秒 (T - {SECONDS_BEFORE})")
    print(f"照片时间戳 (T) 之后: {SECONDS_AFTER} 秒 (T + {SECONDS_AFTER})")
    print(f"总窗口持续时间: {SECONDS_BEFORE + SECONDS_AFTER} 秒")
    print("=" * 66)

    points_lidar_list = []
    total_points_matched = 0
    non_empty_segments = 0

    for info in tqdm(sorted_photo_info, desc="Slicing point cloud"):
        photo_time = info['relative_time']

        # (核心修改) 计算非对称时间窗口
        start_window = photo_time - SECONDS_BEFORE
        end_window = photo_time + SECONDS_AFTER

        # 筛选点云
        bool_mask = (diff_gps_time >= start_window) & (diff_gps_time < end_window)
        points_in_segment = lidars[bool_mask]

        # 更新统计
        num_points = len(points_in_segment)
        if num_points > 0:
            non_empty_segments += 1
            total_points_matched += num_points
        points_lidar_list.append(points_in_segment)

    # --- 5. 保存结果并生成总结报告 ---
    print("\n划分完成，正在保存结果...")
    with open(str(las_output_path / 'points_lidar_list.pkl'), "wb") as file:
        pickle.dump(points_lidar_list, file)
    print(f"点云分段列表已保存至: {str(las_output_path / 'points_lidar_list.pkl')}")

    total_segments = len(points_lidar_list)
    print("\n" + "=" * 25 + " 最终匹配结果总结 " + "=" * 25)
    print(f"总共为 {total_segments} 张照片生成了时间分段。")
    print(f"其中有 {non_empty_segments} 个分段成功匹配到了点云 (非空)。")
    print(f"总共匹配到的点云数量为: {total_points_matched}")
    if total_segments > 0:
        avg_points = total_points_matched / non_empty_segments if non_empty_segments > 0 else 0
        print(f"平均每个非空分段包含 {avg_points:.1f} 个点。")

    if non_empty_segments > total_segments * 0.8 and total_points_matched > 1000:
        print("✅ 匹配情况良好！大部分时间段都找到了对应的点云。")
    elif non_empty_segments > 0:
        print("⚠️ 注意：部分时间段未能匹配到点云，请检查照片和LiDAR数据的时间覆盖范围是否完全重合。")
    else:
        print("❌ 严重错误：所有时间段都未能匹配到点云！请重新检查时间戳转换逻辑或数据源。")
    print("=" * 70)
    print('脚本执行完毕。')


if __name__ == '__main__':
    main(_get_opts())

# SBMU    python 1_process_each_line_las_1.5.py --dataset_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output --las_path F:\download\SMBU2\SMBU1\lidars\terra_las\cloud_merged.las --las_output_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output --num_val 10

# SZIIT    python 1_process_each_line_las_1.8.py --dataset_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output --las_path H:\BaiduNetdiskDownload\LiDAR_Raw\SZIIT\SZTIH018\lidars\terra_las\cloud_merged.las --las_output_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output --num_val 10

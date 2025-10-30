# #
# # from plyfile import PlyData
# # import numpy as np
# # from tqdm import tqdm
# #
# # import laspy
# # import pyproj
# # import torch
# # import datetime
# # from datetime import timedelta
# # from datetime import datetime
# # from pathlib import Path
# # from argparse import Namespace
# # import configargparse
# #
# # import pickle
# # import os
# # import sys
# # current_dir = os.path.dirname(os.path.abspath(__file__))
# # # 上一级目录: .../dji
# # parent_dir = os.path.dirname(current_dir)
# # # 上上一级目录: .../Aerial_lifting_early
# # grandparent_dir = os.path.dirname(parent_dir)
# #
# # # 将 'Aerial_lifting_early' 目录添加到 sys.path
# # if grandparent_dir not in sys.path:
# #     sys.path.append(grandparent_dir)
# # from dji.opt import _get_opts
# #
# # def main(hparams):
# #     dataset_path = Path(hparams.dataset_path)
# #     train_path_candidates = sorted(list((dataset_path / 'train' / 'image_metadata').iterdir()))
# #     train_paths = [train_path_candidates[i] for i in range(0, len(train_path_candidates))]
# #     val_path_candidates = sorted(list((dataset_path / 'val' / 'image_metadata').iterdir()))
# #     val_paths = [val_path_candidates[i] for i in range(0, len(val_path_candidates))]
# #     total_num=len(train_paths+val_paths)
# #     print("process .las file, wait a minute...")
# #     # Load the LAS file
# #     in_file = laspy.file.File(hparams.las_path, mode='r')
# #
# #     x, y, z = in_file.x, in_file.y, in_file.z
# #     r = (in_file.red / 65536.0 * 255).astype(np.uint8)
# #     g = (in_file.green / 65536.0 * 255).astype(np.uint8)
# #     b = (in_file.blue / 65536.0 * 255).astype(np.uint8)
# #     lidars = np.array(list(zip(x, y, z, r, g, b)))
# #
# #     las_output_path = Path(hparams.las_output_path)
# #     if not os.path.exists(hparams.las_output_path):
# #         las_output_path.mkdir(parents=True)
# #     # np.save(hparams.las_output_path +'lidars', lidars)
# #     # intensity=in_file.intensity
# #
# #     # diff_list = []
# #     #
# #     # for i in tqdm(range(total_num)):
# #     # # for i in tqdm(range(hparams.start, hparams.end), desc="calculate GPS time"):
# #     #     if i % int(total_num / hparams.num_val) == 0:
# #     #         split_dir = dataset_path / 'val'
# #     #     else:
# #     #         split_dir = dataset_path / 'train'
# #     #
# #     #     # image_metadata = torch.load(str(split_dir / 'image_metadata' / '{0:06d}.pt'.format(i)))
# #     #     image_metadata = torch.load(str(split_dir / 'image_metadata' / '{0:06d}.pt'.format(i)), weights_only=False)
# #     #     time = image_metadata['meta_tags']['EXIF DateTimeOriginal'].values
# #     #     # date_str, time_str = time.split()
# #     #     # year, month, day = date_str.split(":")
# #     #     # hour, minute, second = time_str.split(":")
# #     #     # current_seconds = int(hour) * 3600 + int(minute) * 60 + int(second)
# #     #     import datetime
# #     #     import time as time_module
# #     #
# #     #     # ...
# #     #     # 原始的EXIF时间字符串，例如 '2024:02:28 14:45:10'
# #     #     time_str_from_exif = image_metadata['meta_tags']['EXIF DateTimeOriginal'].values
# #     #
# #     #     # 1. 将EXIF字符串解析为datetime对象
# #     #     #    注意：我们假设EXIF时间是你所在的本地时区时间（例如UTC+8）
# #     #     dt_local = datetime.datetime.strptime(time_str_from_exif, '%Y:%m:%d %H:%M:%S')
# #     #
# #     #     # 2. 将本地时间datetime对象转换为Unix时间戳 (从1970年1月1日UTC开始的秒数)
# #     #     #    .timestamp()会自动考虑你的电脑系统设置的时区
# #     #     unix_timestamp = dt_local.timestamp()
# #     #
# #     #     # 用新的时间戳替换原来的 current_seconds
# #     #     current_seconds = unix_timestamp
# #     #
# #     #     # ====== 添加调试代码，确认转换结果 ======
# #     #     if i < 5:
# #     #         print(f"图片 {i}: EXIF时间 = '{time_str_from_exif}', 转换后的Unix时间戳 = {unix_timestamp}")
# #     #         # ========================================
# #     #
# #     #         # ...
# #     #
# #     #     if i == hparams.start:
# #     #         origin_second = current_seconds
# #     #         diff=0
# #     #         diff_list.append(0)
# #     #     else:
# #     #         previous_diff = diff
# #     #         diff = current_seconds-origin_second
# #     #         # diff_list.append(diff-previous_diff)
# #     #         diff_list.append(diff)
# #     #
# #     # diff_gps_time = np.array(in_file.gps_time - in_file.gps_time[0])
# #     import datetime
# #
# #     # --- 新增的排序逻辑 ---
# #     print("正在加载所有图像元数据并按EXIF时间排序...")
# #     all_metadata_paths = train_paths + val_paths
# #     all_metadata = []
# #     for path in tqdm(all_metadata_paths, desc="Loading metadata"):
# #         # 加载元数据，同时保留它的原始路径以便后续使用
# #         metadata = torch.load(path, weights_only=False)
# #         all_metadata.append({'data': metadata, 'path': path})
# #
# #     # 关键一步：根据EXIF中的 'DateTimeOriginal' 字段进行排序
# #     all_metadata.sort(key=lambda x: x['data']['meta_tags']['EXIF DateTimeOriginal'].values)
# #
# #     print("排序完成，按正确的时间顺序计算时间差...")
# #     diff_list = []
# #     origin_timestamp = None
# #
# #     # 用排序后的列表来计算时间差
# #     for item in tqdm(all_metadata, desc="Calculating time differences"):
# #         metadata = item['data']
# #         time_str_from_exif = metadata['meta_tags']['EXIF DateTimeOriginal'].values
# #
# #         # 将EXIF时间字符串转换为Unix时间戳
# #         dt_local = datetime.datetime.strptime(time_str_from_exif, '%Y:%m:%d %H:%M:%S')
# #         unix_timestamp = dt_local.timestamp()
# #
# #         if origin_timestamp is None:
# #             origin_timestamp = unix_timestamp
# #
# #         # 计算相对于第一张照片的时间差（秒）
# #         time_difference = unix_timestamp - origin_timestamp
# #         diff_list.append(time_difference)
# #
# #     # 检查一下排序后的时间差，它现在应该是单调递增的
# #     print(f"[调试] 排序后的前10个时间差: {[f'{d:.2f}s' for d in diff_list[:10]]}")
# #     # --- 排序逻辑结束 ---
# #
# #     # 原始脚本中的后续部分可以保持不变...
# #     diff_gps_time = np.array(in_file.gps_time - in_file.gps_time[0])
# #     # points_lidar_list = []
# #     #
# #     # for j in tqdm(range(1, len(diff_list)), desc="save lidars into file"):
# #     #     bool_mask = (diff_gps_time >= int(diff_list[j-1])) * (diff_gps_time < int(diff_list[j]))
# #     #     points_lidar_list.append(lidars[bool_mask])
# #     #
# #     # bool_mask = diff_gps_time >= int(diff_list[-1])
# #     # points_lidar_list.append(lidars[bool_mask])
# #
# #     points_lidar_list = []
# #
# #     # ====== 新增：初始化统计变量 ======
# #     total_points_matched = 0
# #     non_empty_segments = 0
# #
# #     print("\n[调试] 开始按时间分段并统计匹配点数...")
# #     for j in tqdm(range(1, len(diff_list)), desc="save lidars into file"):
# #         # 时间区间的开始和结束（相对于第一个点/第一张照片）
# #         start_time_diff = int(diff_list[j - 1])
# #         end_time_diff = int(diff_list[j])
# #
# #         # 筛选出在该时间区间内的点
# #         bool_mask = (diff_gps_time >= start_time_diff) * (diff_gps_time < end_time_diff)
# #
# #         # 获取匹配到的点
# #         points_in_segment = lidars[bool_mask]
# #         num_points_in_segment = len(points_in_segment)
# #
# #         # 实时打印前几个分段的匹配情况
# #         if j <= 5:
# #             print(
# #                 f"  [调试] 分段 {j - 1}->{j} (时间差 {start_time_diff}s - {end_time_diff}s): 匹配到 {num_points_in_segment} 个点。")
# #
# #         # 更新统计数据
# #         if num_points_in_segment > 0:
# #             non_empty_segments += 1
# #             total_points_matched += num_points_in_segment
# #
# #         points_lidar_list.append(points_in_segment)
# #     # 处理最后一个时间段
# #     start_time_diff = int(diff_list[-1])
# #     bool_mask = diff_gps_time >= start_time_diff
# #     points_in_segment = lidars[bool_mask]
# #     num_points_in_segment = len(points_in_segment)
# #     print(f"  [调试] 最后一个分段 (时间差 > {start_time_diff}s): 匹配到 {num_points_in_segment} 个点。")
# #     if num_points_in_segment > 0:
# #         non_empty_segments += 1
# #         total_points_matched += num_points_in_segment
# #     points_lidar_list.append(points_in_segment)
# #
# #     with open(str(las_output_path / 'points_lidar_list.pkl'), "wb") as file:
# #         pickle.dump(points_lidar_list, file)
# #     print(f"lidars save at {str(las_output_path / 'points_lidar_list.pkl')}")
# #
# #     # ====== 新增：在脚本末尾打印最终总结 ======
# #     print("\n" + "=" * 25 + " 匹配结果验证 " + "=" * 25)
# #     total_segments = len(points_lidar_list)
# #     print(f"总共为 {total_segments} 张照片（时间点）生成了 {total_segments} 个时间分段。")
# #     print(f"其中有 {non_empty_segments} 个分段成功匹配到了点云 (非空)。")
# #     print(f"总共匹配到的点云数量为: {total_points_matched}")
# #
# #     if non_empty_segments > total_segments * 0.8 and total_points_matched > 1000:
# #         print("✅ 匹配情况良好！大部分时间段都找到了对应的点云。")
# #     elif non_empty_segments > 0:
# #         print("⚠️ 注意：部分时间段未能匹配到点云，请检查照片和LiDAR数据的时间覆盖范围是否完全重合。")
# #     else:
# #         print("❌ 严重错误：所有时间段都未能匹配到点云！请重新检查时间戳转换逻辑或数据源。")
# #     print("=" * 70)
# #
# #     print('done')
# #
# #     with open(str(las_output_path / 'points_lidar_list.pkl'), "wb") as file:
# #         pickle.dump(points_lidar_list, file)
# #     print(f"lidars save at {str(las_output_path / 'points_lidar_list.pkl')}")
# #
# #     print('done')
# #
# # if __name__ == '__main__':
# #     main(_get_opts())
# #
# #  #python 1_process_each_line_las.py --dataset_path F:\download\SMBU2\output --las_path F:\download\SMBU2\lidars\terra_las\cloud_merged.las --las_output_path F:\download\SMBU2\output\lidar_processed --num_val 10
#
# # 1_process_each_line_las_DEBUG.py
#
# from plyfile import PlyData
# # 脚本功能：
# # 1. 忽略.las文件中损坏的GPS时间戳。
# # 2. 根据用户在下方填写的SDK日志中的真实起止时间，重建LiDAR点云的正确时间戳。
# # 3. 加载所有照片的元数据，并严格按照EXIF拍摄时间进行排序。
# # 4. 将照片时间转换为标准的Unix时间戳。
# # 5. 使用稳健的时间窗口匹配逻辑，将点云分段并关联到每张照片。
# # 6. 输出分段后的点云列表(points_lidar_list.pkl)和排序后的照片ID列表(sorted_image_ids.pkl)。
#
# import numpy as np
# from tqdm import tqdm
# import laspy
# import torch
# import datetime
# from pathlib import Path
# import pickle
# import os
# import sys
#
# # --- 项目路径设置 (请根据你的结构确认) ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# grandparent_dir = os.path.dirname(parent_dir)
# if grandparent_dir not in sys.path:
#     sys.path.append(grandparent_dir)
# from dji.opt import _get_opts
#
#
# # ----------------------------------------
#
# def main(hparams):
#     # ==============================================================================
#     # ==                            用户需手动修改区域                            ==
#     # ==============================================================================
#     # 请从你的SDK日志文件中找到LiDAR扫描的准确开始和结束时间，并填入下方
#     # 格式为标准的Unix时间戳 (一个以1...开头的大数字)
#
#     LIDAR_START_UNIX_TIME = 1700365245.262816
#     LIDAR_END_UNIX_TIME = 1700365325.179499
#
#     # ==============================================================================
#
#     # --- 1. 加载LiDAR点云几何数据 ---
#     dataset_path = Path(hparams.dataset_path)
#     las_output_path = Path(hparams.las_output_path)
#     las_output_path.mkdir(parents=True, exist_ok=True)
#
#     print("正在加载LAS文件几何数据 (将忽略其内部的错误时间戳)...")
#     # in_file = laspy.read(hparams.las_path)
#     in_file = laspy.file.File(hparams.las_path, mode='r')
#     x, y, z = in_file.x, in_file.y, in_file.z
#     r = (in_file.red / 65536.0 * 255).astype(np.uint8)
#     g = (in_file.green / 65536.0 * 255).astype(np.uint8)
#     b = (in_file.blue / 65536.0 * 255).astype(np.uint8)
#     lidars = np.array(list(zip(x, y, z, r, g, b)))
#     print("LiDAR几何数据加载完成。")
#
#     # --- 2. 重建LiDAR的正确时间戳 ---
#     num_lidar_points = len(in_file.points)
#     print(f"检测到 {num_lidar_points} 个LiDAR点。")
#     print("正在根据SDK日志的起止时间，线性重建每个点的正确Unix时间戳...")
#
#     reconstructed_lidar_unix_times = np.linspace(
#         LIDAR_START_UNIX_TIME,
#         LIDAR_END_UNIX_TIME,
#         num=num_lidar_points
#     )
#     print("LiDAR时间戳重建完成！")
#
#     # --- 3. 加载并按EXIF时间排序照片元数据 ---
#     train_path_candidates = sorted(list((dataset_path / 'train' / 'image_metadata').iterdir()))
#     val_path_candidates = sorted(list((dataset_path / 'val' / 'image_metadata').iterdir()))
#     all_metadata_paths = train_path_candidates + val_path_candidates
#
#     print("\n正在加载所有图像元数据并按EXIF拍摄时间排序...")
#     all_metadata = []
#     for path in tqdm(all_metadata_paths, desc="Loading metadata"):
#         metadata = torch.load(path, weights_only=False)
#         image_id = path.stem  # 提取文件名作为ID, e.g., '000001'
#         all_metadata.append({'data': metadata, 'id': image_id})
#
#     # 关键一步：根据EXIF中的 'DateTimeOriginal' 字段进行排序
#     all_metadata.sort(key=lambda x: x['data']['meta_tags']['EXIF DateTimeOriginal'].values)
#     print("照片元数据排序完成。")
#
#     # --- 4. 将照片时间转换为Unix时间戳 ---
#     photo_unix_timestamps = []
#     sorted_image_ids = []
#     for item in all_metadata:
#         time_str = item['data']['meta_tags']['EXIF DateTimeOriginal'].values
#         # 注意: .timestamp() 会使用你当前电脑的本地时区设置来转换。
#         # 请确保你电脑的时区与无人机飞行作业时的时区设置一致。
#         dt_local = datetime.datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
#         photo_unix_timestamps.append(dt_local.timestamp())
#         sorted_image_ids.append(item['id'])
#     print(f"照片时间戳转换完成。共 {len(photo_unix_timestamps)} 张照片按时间排序。")
#     print(f"照片时间范围 (Unix): {photo_unix_timestamps[0]:.2f}  到  {photo_unix_timestamps[-1]:.2f}")
#
#     # --- 5. 执行匹配 ---
#     points_lidar_list = []
#     print("\n开始使用重建的LiDAR时间和照片Unix时间进行匹配...")
#
#     for i in tqdm(range(len(photo_unix_timestamps)), desc="Matching photos to point cloud"):
#         # 定义每个照片的时间窗口：从上一个照片和当前照片的中点，到当前和下一个照片的中点
#         if i == 0:
#             # 第一张照片的窗口从LiDAR开始时间算起
#             start_time = LIDAR_START_UNIX_TIME
#             end_time = (photo_unix_timestamps[i] + photo_unix_timestamps[i + 1]) / 2
#         elif i == len(photo_unix_timestamps) - 1:
#             # 最后一张照片的窗口直到LiDAR结束时间
#             start_time = (photo_unix_timestamps[i - 1] + photo_unix_timestamps[i]) / 2
#             end_time = LIDAR_END_UNIX_TIME
#         else:
#             start_time = (photo_unix_timestamps[i - 1] + photo_unix_timestamps[i]) / 2
#             end_time = (photo_unix_timestamps[i] + photo_unix_timestamps[i + 1]) / 2
#
#         # 筛选出在该时间窗口内的点云
#         mask = (reconstructed_lidar_unix_times >= start_time) & (reconstructed_lidar_unix_times < end_time)
#         points_in_segment = lidars[mask]
#         points_lidar_list.append(points_in_segment)
#
#     # --- 6. 保存结果并生成总结报告 ---
#     print("\n匹配完成，正在保存结果...")
#     with open(las_output_path / 'points_lidar_list.pkl', "wb") as file:
#         pickle.dump(points_lidar_list, file)
#     # 保存排序后的ID列表，这对于脚本2至关重要！
#     with open(las_output_path / 'sorted_image_ids.pkl', 'wb') as f:
#         pickle.dump(sorted_image_ids, f)
#     print(f"结果已保存到目录: {las_output_path}")
#
#     # 生成最终总结
#     total_segments = len(points_lidar_list)
#     non_empty_segments = 0
#     total_points_matched = 0
#     for segment in points_lidar_list:
#         if len(segment) > 0:
#             non_empty_segments += 1
#             total_points_matched += len(segment)
#
#     print("\n" + "=" * 25 + " 最终匹配结果总结 " + "=" * 25)
#     print(f"总共为 {total_segments} 张照片生成了时间分段。")
#     print(f"其中有 {non_empty_segments} 个分段成功匹配到了点云 (非空)。")
#     print(f"总共匹配到的点云数量为: {total_points_matched}")
#
#     if non_empty_segments > total_segments * 0.8 and total_points_matched > 1000:
#         print("✅ 匹配情况良好！大部分时间段都找到了对应的点云。")
#     elif non_empty_segments > 0:
#         print("⚠️ 注意：部分时间段未能匹配到点云，这可能是因为照片拍摄时长超出了LiDAR扫描时长，属于正常现象。")
#     else:
#         print("❌ 严重错误：所有时间段都未能匹配到点云！请检查SDK日志中的起止时间是否填写正确。")
#     print("=" * 70)
#
#     print('脚本执行完毕。')
#
#
# if __name__ == '__main__':
#     main(_get_opts())
#
# 脚本功能：
# 1. 加载所有照片元数据，并严格按照EXIF拍摄时间进行排序。
# 2. 将照片时间转换为标准的Unix时间戳。
# 3. (核心修改) 对于每一张照片，读取其'曝光时间'。
# 4. (核心修改) 以每张照片的精确时间戳为中心，以其曝光时间为宽度，创建一个极窄的时间窗口。
# 5. 使用这个精确的时间窗口，从总点云中筛选出与该照片在物理上最匹配的点云。
# 6. 输出分段后的点云列表 (points_lidar_list.pkl)。

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


# --- 新增辅助函数，用于解析曝光时间字符串 ---
def parse_exposure_time(time_str):
    """将 '1/2000' 这样的字符串转换为秒(float)"""
    try:
        if '/' in time_str:
            num, den = map(float, time_str.split('/'))
            return num / den
        else:
            return float(time_str)
    except (ValueError, ZeroDivisionError):
        # 如果格式不正确或分母为0，返回一个默认的极小值
        print(f"警告: 无法解析曝光时间 '{time_str}', 将使用默认值 0.001 秒。")
        return 0.001


def main(hparams):
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

    # 计算LiDAR点云的相对时间戳 (从0开始的秒数)
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

    # 关键一步：根据EXIF中的 'DateTimeOriginal' 字段进行排序
    all_metadata.sort(key=lambda x: x['data']['meta_tags']['EXIF DateTimeOriginal'].values)
    print("照片元数据排序完成。")

    # --- 3. (核心修改) 提取精确时间戳和曝光时间 ---
    print("\n正在提取每张照片的精确时间和曝光时间...")
    sorted_photo_info = []
    origin_timestamp = None

    for item in tqdm(all_metadata, desc="Extracting photo info"):
        metadata = item['data']

        # 提取时间戳
        time_str = metadata['meta_tags']['EXIF DateTimeOriginal'].values
        dt_local = datetime.datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
        unix_timestamp = dt_local.timestamp()

        if origin_timestamp is None:
            origin_timestamp = unix_timestamp

        # 计算相对于第一张照片的时间差（秒）
        time_difference = unix_timestamp - origin_timestamp

        # 提取曝光时间
        # ==============================================================================
        #  == 请您确认这里的键名是否正确，我假设为 'EXIF ExposureTime' ==
        # ==============================================================================
        exposure_time_key = 'EXIF ExposureTime'
        if exposure_time_key in metadata['meta_tags']:
            exposure_time_str = metadata['meta_tags'][exposure_time_key].values
            exposure_time_sec = parse_exposure_time(exposure_time_str)
        else:
            print(f"警告: 在元数据中未找到键 '{exposure_time_key}'，将使用默认值 0.001 秒。")
            exposure_time_sec = 0.001  # 提供一个默认值
        # ==============================================================================

        sorted_photo_info.append({
            "relative_time": time_difference,
            "exposure_time": exposure_time_sec
        })

    print(f"[调试] 已提取 {len(sorted_photo_info)} 张照片的信息。")
    print(
        f"      第一张照片的相对时间: {sorted_photo_info[0]['relative_time']:.3f}s, 曝光时间: {sorted_photo_info[0]['exposure_time']:.5f}s")
    print(
        f"      最后一张照片的相对时间: {sorted_photo_info[-1]['relative_time']:.3f}s, 曝光时间: {sorted_photo_info[-1]['exposure_time']:.5f}s")

    # --- 4. (核心修改) 按曝光时间窗口进行点云划分 ---
    print("\n正在按每张照片的曝光时间窗口进行点云划分...")
    points_lidar_list = []
    total_points_matched = 0
    non_empty_segments = 0

    for info in tqdm(sorted_photo_info, desc="Slicing point cloud"):
        # 计算精确的时间窗口
        photo_time = info['relative_time']
        exposure = info['exposure_time']

        start_window = photo_time - (exposure / 2.0)
        end_window = photo_time + (exposure / 2.0)

        # 筛选出在该精确时间窗口内的点
        bool_mask = (diff_gps_time >= start_window) & (diff_gps_time < end_window)
        points_in_segment = lidars[bool_mask]

        # 更新统计数据
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

    # 生成最终总结
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
#python 1_process_each_line_las.py --dataset_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output --las_path F:\download\SMBU2\SMBU1\lidars\terra_las\cloud_merged.las --las_output_path F:\download\SMBU2\output\lidar_processed --num_val 10
#F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output
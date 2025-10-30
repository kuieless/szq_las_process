# # #
# # #
# # #
# # # import laspy
# # # import numpy as np
# # # import datetime
# # # import pytz
# # # import matplotlib.pyplot as plt
# # #
# # # # --- 1. 用户配置 ---
# # #
# # # # 请将这里替换为您的 .las 文件路径
# # # las_file_path = r"F:\download\SMBU2\SMBU1\lidars\terra_las\cloud_merged.las"
# # #
# # # # 您提供的航班时间分段 (UTC+8)
# # # flight_segments_str = [
# # #     {"name": "航班 1", "start": "2023-11-16 10:53:24", "end": "2023-11-16 10:54:45"},
# # #     {"name": "航班 2", "start": "2023-11-16 10:57:03", "end": "2023-11-16 10:58:25"},
# # #     {"name": "航班 3", "start": "2023-11-16 11:00:55", "end": "2023-11-16 11:01:27"},
# # #     {"name": "航班 4", "start": "2023-11-16 11:03:41", "end": "2023-11-16 11:05:00"},
# # # ]
# # #
# # #
# # # # --- 2. 辅助函数和数据转换 ---
# # #
# # # def to_unix_timestamp(dt_str):
# # #     """将 UTC+8 时间字符串转换为 Unix 时间戳"""
# # #     tz = pytz.timezone('Asia/Singapore')  # UTC+8
# # #     dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
# # #     dt_aware = tz.localize(dt)
# # #     return dt_aware.timestamp()
# # #
# # #
# # # # 将航班时间转换为 Unix 时间戳
# # # flight_segments_unix = [
# # #     {
# # #         "name": s["name"],
# # #         "start": to_unix_timestamp(s["start"]),
# # #         "end": to_unix_timestamp(s["end"]),
# # #     }
# # #     for s in flight_segments_str
# # # ]
# # #
# # # print("--- 航班时间段 (Unix 时间戳) ---")
# # # for s in flight_segments_unix:
# # #     print(f"{s['name']}: {s['start']} -> {s['end']}")
# # # print("-" * 30)
# # #
# # # # --- 3. 主逻辑 ---
# # #
# # # try:
# # #     print(f"正在读取 LAS 文件: {las_file_path}")
# # #     with laspy.open(las_file_path) as f:
# # #         las = f.read()
# # #
# # #         # 提取 GPS 时间数据
# # #         gps_times = las.gps_time
# # #
# # #         # 找到 LAS 文件中最早和最晚的时间点
# # #         min_las_time = np.min(gps_times)
# # #         max_las_time = np.max(gps_times)
# # #
# # #         print(f"LAS 文件原始 GPS 时间范围: {min_las_time} -> {max_las_time}")
# # #
# # #         # 获取第一个航班的开始时间
# # #         first_flight_start_time = flight_segments_unix[0]["start"]
# # #
# # #         # 根据假设，计算时间偏移量
# # #         time_offset = first_flight_start_time - min_las_time
# # #         print(f"计算出的时间偏移量: {time_offset:.4f} 秒")
# # #
# # #         # 校准所有 LAS 点的时间
# # #         adjusted_gps_times = gps_times + time_offset
# # #
# # #         print("\n--- 时间分布统计 ---")
# # #         total_points_in_flights = 0
# # #         for segment in flight_segments_unix:
# # #             # 使用 NumPy 进行高效筛选和计数
# # #             points_in_segment = np.sum(
# # #                 (adjusted_gps_times >= segment["start"]) & (adjusted_gps_times <= segment["end"])
# # #             )
# # #             total_points_in_flights += points_in_segment
# # #             print(f"在 '{segment['name']}' 时间段内找到 {points_in_segment:,} 个点")
# # #
# # #         print(f"\n总点数: {len(las.points):,}")
# # #         print(f"所有航班段内的总点数: {total_points_in_flights:,}")
# # #
# # #         # --- 4. 可视化 ---
# # #         print("\n正在生成时间分布直方图...")
# # #         plt.figure(figsize=(15, 7))
# # #
# # #         # 绘制直方图
# # #         plt.hist(adjusted_gps_times, bins=1000, label="LAS 点数据分布")
# # #
# # #         # 用垂直线标记航班起止时间
# # #         colors = ['green', 'red', 'purple', 'orange']
# # #         for i, segment in enumerate(flight_segments_unix):
# # #             color = colors[i % len(colors)]
# # #             plt.axvline(segment['start'], color=color, linestyle='--', label=f"{segment['name']} Start")
# # #             plt.axvline(segment['end'], color=color, linestyle=':', label=f"{segment['name']} End")
# # #
# # #         plt.title("LAS 点云时间分布与航班分段对比图")
# # #         plt.xlabel("校准后的时间 (Unix Timestamp)")
# # #         plt.ylabel("点数量")
# # #         plt.legend()
# # #         plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# # #
# # #         # 格式化 X 轴标签为可读时间
# # #         formatter = plt.FuncFormatter(
# # #             lambda x, pos: datetime.datetime.fromtimestamp(x, tz=pytz.timezone('Asia/Singapore')).strftime('%H:%M:%S'))
# # #         plt.gca().xaxis.set_major_formatter(formatter)
# # #         plt.xticks(rotation=45)
# # #         plt.tight_layout()
# # #
# # #         plt.show()
# # #
# # # except FileNotFoundError:
# # #     print(f"错误: 文件未找到 at '{las_file_path}'")
# # # except Exception as e:
# # #     print(f"处理过程中发生错误: {e}")
# #
# #
# # import laspy
# # import numpy as np
# # import datetime
# # import pytz
# # import matplotlib.pyplot as plt
# # import csv
# # import math
# #
# # # --- 1. 用户配置 ---
# #
# # # 请将这里替换为您的 .las 文件路径
# # las_file_path = r"F:\download\SMBU2\SMBU1\lidars\terra_las\cloud_merged.las"
# #
# # # 您提供的航班时间分段 (UTC+8)
# # flight_segments_str = [
# #     {"name": "航班 1", "start": "2023-11-16 10:53:24", "end": "2023-11-16 10:54:45"},
# #     {"name": "航班 2", "start": "2023-11-16 10:57:03", "end": "2023-11-16 10:58:25"},
# #     {"name": "航班 3", "start": "2023-11-16 11:00:55", "end": "2023-11-16 11:01:27"},
# #     {"name": "航班 4", "start": "2023-11-16 11:03:41", "end": "2023-11-16 11:05:00"},
# # ]
# #
# # # --- 新增功能: 定义输出的日志文件名 ---
# # output_csv_file = "flight_time_log_detailed.csv"
# #
# #
# # # --- 2. 辅助函数和数据转换 ---
# #
# # def to_unix_timestamp(dt_str):
# #     """将 UTC+8 时间字符串转换为 Unix 时间戳"""
# #     tz = pytz.timezone('Asia/Singapore')  # UTC+8
# #     dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
# #     dt_aware = tz.localize(dt)
# #     return dt_aware.timestamp()
# #
# #
# # # --- 新增功能: 判断时间戳状态的函数 ---
# # def get_flight_status(timestamp, segments):
# #     """根据时间戳判断其所属的航班状态"""
# #     for i, seg in enumerate(segments):
# #         if seg['start'] <= timestamp <= seg['end']:
# #             return seg['name']
# #         # 检查是否在两个航班之间的间隔期
# #         if i > 0 and segments[i - 1]['end'] < timestamp < seg['start']:
# #             return f"间隔于 {segments[i - 1]['name'].split(' ')[1]} & {seg['name'].split(' ')[1]}"
# #     return "任务外"
# #
# #
# # # 将航班时间转换为 Unix 时间戳
# # flight_segments_unix = [
# #     {
# #         "name": s["name"],
# #         "start": to_unix_timestamp(s["start"]),
# #         "end": to_unix_timestamp(s["end"]),
# #     }
# #     for s in flight_segments_str
# # ]
# #
# # print("--- 航班时间段 (Unix 时间戳) ---")
# # for s in flight_segments_unix:
# #     print(f"{s['name']}: {s['start']} -> {s['end']}")
# # print("-" * 30)
# #
# # # --- 3. 主逻辑 ---
# #
# # try:
# #     print(f"正在读取 LAS 文件: {las_file_path}")
# #     with laspy.open(las_file_path) as f:
# #         las = f.read()
# #
# #     gps_times = las.gps_time
# #     min_las_time, max_las_time = np.min(gps_times), np.max(gps_times)
# #     print(f"LAS 文件原始 GPS 时间范围: {min_las_time} -> {max_las_time}")
# #
# #     first_flight_start_time = flight_segments_unix[0]["start"]
# #     time_offset = first_flight_start_time - min_las_time
# #     print(f"计算出的时间偏移量: {time_offset:.4f} 秒")
# #     adjusted_gps_times = gps_times + time_offset
# #
# #     print("\n--- 时间分布统计 ---")
# #     total_points_in_flights = 0
# #     for segment in flight_segments_unix:
# #         points_in_segment = np.sum(
# #             (adjusted_gps_times >= segment["start"]) & (adjusted_gps_times <= segment["end"])
# #         )
# #         total_points_in_flights += points_in_segment
# #         print(f"在 '{segment['name']}' 时间段内找到 {points_in_segment:,} 个点")
# #
# #     print(f"\n总点数: {len(las.points):,}")
# #     print(f"所有航班段内的总点数: {total_points_in_flights:,}")
# #
# #     # --- 新增功能: 生成详细的CSV日志文件 ---
# #     print("\n--- 正在生成详细CSV日志 ---")
# #
# #     start_unix_second = math.floor(np.min(adjusted_gps_times))
# #     end_unix_second = math.ceil(np.max(adjusted_gps_times))
# #
# #     # 使用 np.histogram 高效计算每秒的点数
# #     bins = np.arange(start_unix_second, end_unix_second + 2)
# #     point_counts, _ = np.histogram(adjusted_gps_times, bins=bins)
# #
# #     with open(output_csv_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
# #         log_writer = csv.writer(csvfile)
# #         header = ["日期时间 (UTC+8)", "校准后_Unix时间戳", "原始_LAS_GPS时间戳", "状态", "该秒内点数"]
# #         log_writer.writerow(header)
# #
# #         for i, count in enumerate(point_counts):
# #             current_unix_ts = start_unix_second + i
# #             original_gps_ts = current_unix_ts - time_offset
# #             human_readable_time = datetime.datetime.fromtimestamp(current_unix_ts,
# #                                                                   tz=pytz.timezone('Asia/Singapore')).strftime(
# #                 '%Y-%m-%d %H:%M:%S')
# #             status = get_flight_status(current_unix_ts, flight_segments_unix)
# #             row = [human_readable_time, current_unix_ts, f"{original_gps_ts:.6f}", status, count]
# #             log_writer.writerow(row)
# #
# #     print(f"🎉 成功！详细日志已保存至: {output_csv_file}")
# #     # ---------------------------------------------
# #
# #     # --- 4. 可视化 ---
# #     print("\n正在生成时间分布直方图...")
# #     plt.figure(figsize=(15, 7))
# #
# #     plt.hist(adjusted_gps_times, bins=1000, label="LAS 点数据分布")
# #
# #     colors = ['green', 'red', 'purple', 'orange']
# #     for i, segment in enumerate(flight_segments_unix):
# #         color = colors[i % len(colors)]
# #         plt.axvline(segment['start'], color=color, linestyle='--', label=f"{segment['name']} Start")
# #         plt.axvline(segment['end'], color=color, linestyle=':', label=f"{segment['name']} End")
# #
# #     plt.title("LAS 点云时间分布与航班分段对比图")
# #     plt.xlabel("校准后的时间 (Unix Timestamp)")
# #     plt.ylabel("点数量")
# #     plt.legend()
# #     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# #
# #     formatter = plt.FuncFormatter(
# #         lambda x, pos: datetime.datetime.fromtimestamp(x, tz=pytz.timezone('Asia/Singapore')).strftime('%H:%M:%S'))
# #     plt.gca().xaxis.set_major_formatter(formatter)
# #     plt.xticks(rotation=45)
# #     plt.tight_layout()
# #
# #     plt.show()
# #
# # except FileNotFoundError:
# #     print(f"错误: 文件未找到 at '{las_file_path}'")
# # except Exception as e:
# #     print(f"处理过程中发生错误: {e}")


# import laspy
# import numpy as np
# import datetime
# import pytz
# import matplotlib.pyplot as plt
# import csv
# import math

# # --- 1. 用户配置 ---

# # 请将这里替换为您的 .las 文件路径
# las_file_path = r"F:\download\SMBU2\SMBU1\lidars\terra_las\cloud_merged.las"

# # 您提供的航班时间分段 (UTC+8)
# flight_segments_str = [
#     {"name": "航班 1", "start": "2023-11-16 10:53:24", "end": "2023-11-16 10:54:45"},
#     {"name": "航班 2", "start": "2023-11-16 10:57:03", "end": "2023-11-16 10:58:25"},
#     {"name": "航班 3", "start": "2023-11-16 11:00:55", "end": "2023-11-16 11:01:27"},
#     {"name": "航班 4", "start": "2023-11-16 11:03:41", "end": "2023-11-16 11:05:00"},
# ]

# # 定义输出的日志文件名
# output_csv_file = "flight_time_log_detailed.csv"


# # --- 2. 辅助函数和数据转换 ---

# def to_unix_timestamp(dt_str):
#     """将 UTC+8 时间字符串转换为 Unix 时间戳"""
#     tz = pytz.timezone('Asia/Singapore')  # UTC+8
#     dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
#     dt_aware = tz.localize(dt)
#     return dt_aware.timestamp()


# def get_flight_status(timestamp, segments):
#     """根据时间戳判断其所属的航班状态"""
#     for i, seg in enumerate(segments):
#         if seg['start'] <= timestamp <= seg['end']:
#             return seg['name']
#         if i > 0 and segments[i - 1]['end'] < timestamp < seg['start']:
#             return f"间隔于 {segments[i - 1]['name'].split(' ')[1]} & {seg['name'].split(' ')[1]}"
#     return "任务外"


# # 将航班时间转换为 Unix 时间戳
# flight_segments_unix = [
#     {
#         "name": s["name"],
#         "start": to_unix_timestamp(s["start"]),
#         "end": to_unix_timestamp(s["end"]),
#     }
#     for s in flight_segments_str
# ]

# print("--- 航班时间段 (Unix 时间戳) ---")
# for s in flight_segments_unix:
#     print(f"{s['name']}: {s['start']} -> {s['end']}")
# print("-" * 30)

# # <--- 新增信息 --->
# print("\n--- 航线时长分析 ---")
# total_flight_duration = 0
# for segment in flight_segments_unix:
#     duration = segment['end'] - segment['start']
#     total_flight_duration += duration
#     print(f"  - {segment['name']} 持续时长: {duration:.2f} 秒")
# print(f"总计有效航线时长 (不含间隔): {total_flight_duration:.2f} 秒")
# print("-" * 30)
# # <--- 新增信息结束 --->


# # --- 3. 主逻辑 ---

# try:
#     print("\n--- LAS文件处理与对齐 ---")
#     print(f"正在读取 LAS 文件: {las_file_path}")
#     with laspy.open(las_file_path) as f:
#         las = f.read()

#     gps_times = las.gps_time
#     min_las_time, max_las_time = np.min(gps_times), np.max(gps_times)

#     # <--- 新增信息 --->
#     las_duration = max_las_time - min_las_time
#     print(f"LAS 文件原始 GPS 时间戳:")
#     print(f"  - 最小值: {min_las_time}")
#     print(f"  - 最大值: {max_las_time}")
#     print(f"  - 总时间跨度: {las_duration:.2f} 秒")
#     # <--- 新增信息结束 --->

#     first_flight_start_time = flight_segments_unix[0]["start"]
#     time_offset = first_flight_start_time - min_las_time
#     print(f"计算出的时间偏移量: {time_offset:.4f} 秒")
#     adjusted_gps_times = gps_times + time_offset

#     print("\n--- 时间分布统计 ---")
#     total_points_in_flights = 0
#     for segment in flight_segments_unix:
#         points_in_segment = np.sum(
#             (adjusted_gps_times >= segment["start"]) & (adjusted_gps_times <= segment["end"])
#         )
#         total_points_in_flights += points_in_segment
#         print(f"在 '{segment['name']}' 时间段内找到 {points_in_segment:,} 个点")

#     print(f"\n总点数: {len(las.points):,}")
#     print(f"所有航班段内的总点数: {total_points_in_flights:,}")

#     # --- 生成详细的CSV日志文件 ---
#     print("\n--- 正在生成详细CSV日志 ---")
#     start_unix_second = math.floor(np.min(adjusted_gps_times))
#     end_unix_second = math.ceil(np.max(adjusted_gps_times))
#     bins = np.arange(start_unix_second, end_unix_second + 2)
#     point_counts, _ = np.histogram(adjusted_gps_times, bins=bins)

#     with open(output_csv_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
#         log_writer = csv.writer(csvfile)
#         header = ["日期时间 (UTC+8)", "校准后_Unix时间戳", "原始_LAS_GPS时间戳", "状态", "该秒内点数"]
#         log_writer.writerow(header)
#         for i, count in enumerate(point_counts):
#             current_unix_ts = start_unix_second + i
#             original_gps_ts = current_unix_ts - time_offset
#             human_readable_time = datetime.datetime.fromtimestamp(current_unix_ts,
#                                                                   tz=pytz.timezone('Asia/Singapore')).strftime(
#                 '%Y-%m-%d %H:%M:%S')
#             status = get_flight_status(current_unix_ts, flight_segments_unix)
#             row = [human_readable_time, current_unix_ts, f"{original_gps_ts:.6f}", status, count]
#             log_writer.writerow(row)

#     print(f"🎉 成功！详细日志已保存至: {output_csv_file}")

#     # --- 4. 可视化 ---
#     print("\n正在生成时间分布直方图...")
#     plt.figure(figsize=(15, 7))
#     plt.hist(adjusted_gps_times, bins=1000, label="LAS 点数据分布")
#     colors = ['green', 'red', 'purple', 'orange']
#     for i, segment in enumerate(flight_segments_unix):
#         color = colors[i % len(colors)]
#         plt.axvline(segment['start'], color=color, linestyle='--', label=f"{segment['name']} Start")
#         plt.axvline(segment['end'], color=color, linestyle=':', label=f"{segment['name']} End")
#     plt.title("LAS 点云时间分布与航班分段对比图")
#     plt.xlabel("校准后的时间 (Unix Timestamp)")
#     plt.ylabel("点数量")
#     plt.legend()
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     formatter = plt.FuncFormatter(
#         lambda x, pos: datetime.datetime.fromtimestamp(x, tz=pytz.timezone('Asia/Singapore')).strftime('%H:%M:%S'))
#     plt.gca().xaxis.set_major_formatter(formatter)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# except FileNotFoundError:
#     print(f"错误: 文件未找到 at '{las_file_path}'")
# except Exception as e:
#     print(f"处理过程中发生错误: {e}")


import laspy
import numpy as np
import datetime
import pytz
import matplotlib.pyplot as plt
import csv
import math
import torch
from pathlib import Path
from tqdm import tqdm
import os
import sys
import traceback  # 导入以进行详细错误跟踪

# --- 1. 用户配置 ---

# (!!!) (修改) 请提供包含所有 .las 文件的 *根目录*
las_dir_path = r"/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_lidar/hav"

# (!!!) (新增) 请提供包含 .pt 元数据文件的目录
image_metadata_dir_path = r"/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/hav/image_metadata"

# (!!!) (新增) 自动分段阈值 (秒)
# 如果两张照片的间隔超过 60 秒，则视为一次新的“航班”
FLIGHT_GAP_SECONDS = 60

# 定义输出的日志文件名
output_csv_file = "flight_time_log_detailed.csv"

# 定义时区 (!!!)
# 您的 EXIF 元数据时间戳所在的本地时区
# 'Asia/Singapore' 和 'Asia/Shanghai' 都是 UTC+8
LOCAL_TIMEZONE = 'Asia/Singapore'


# --- 2. 辅助函数 ---

def load_image_timestamps(metadata_dir, gap_seconds_threshold):
    """
    (新增) 
    从 .pt 元数据文件加载所有图像时间戳，
    并自动检测“航班”分段。
    """
    print(f"正在扫描: {metadata_dir}")
    metadata_dir = Path(metadata_dir)
    if not metadata_dir.exists():
        print(f"❌ 严重错误: 找不到图像元数据目录 '{metadata_dir}'")
        return []

    pt_files = sorted(list(metadata_dir.glob('*.pt')))
    if not pt_files:
        print(f"❌ 严重错误: 在 '{metadata_dir}' 中未找到 .pt 元数据文件。")
        return []
    
    print(f"找到 {len(pt_files)} 个图像元数据文件。")
    
    all_image_times = []
    for path in tqdm(pt_files, desc="加载图像元数据 (本地时间)"):
        try:
            metadata = torch.load(path)
            time_str = metadata['meta_tags']['EXIF DateTimeOriginal'].values
            dt_local = datetime.datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
            all_image_times.append(dt_local)
        except Exception as e:
            print(f"  警告: 加载或解析 {path.name} 失败: {e}")
            
    if not all_image_times:
        print("❌ 严重错误: 未能从元数据中解析出任何时间戳。")
        return []

    # (!!!) 核心：自动分段逻辑 (!!!)
    # all_image_times 已经是按文件名（000000.pt, 000001.pt）排序的，
    # 假设文件名顺序 = 时间顺序
    
    segments = []
    if not all_image_times:
        return segments

    gap_threshold = datetime.timedelta(seconds=gap_seconds_threshold)
    
    current_segment_start = all_image_times[0]
    for i in range(1, len(all_image_times)):
        time_gap = all_image_times[i] - all_image_times[i-1]
        
        # 如果发现一个大间隙
        if time_gap > gap_threshold:
            # 1. 关闭上一个分段
            segment_end = all_image_times[i-1]
            segments.append({
                "name": f"航班 {len(segments) + 1}",
                "start_dt": current_segment_start,
                "end_dt": segment_end
            })
            # 2. 开启一个新分段
            current_segment_start = all_image_times[i]

    # 3. 关闭最后一个分段
    segments.append({
        "name": f"航班 {len(segments) + 1}",
        "start_dt": current_segment_start,
        "end_dt": all_image_times[-1]
    })

    print(f"--- 自动检测到 {len(segments)} 个航班分段 (基于 {gap_seconds_threshold} 秒间隔) ---")
    return segments


def load_multiple_las_files(las_dir):
    """
    (新增)
    递归加载一个目录下的所有 .las/.laz 文件，
    并合并它们的 gps_time 和总点数。
    """
    las_dir = Path(las_dir)
    if not las_dir.exists():
        print(f"❌ 严重错误: 找不到 LAS 目录 '{las_dir}'")
        return None, 0
        
    las_files = sorted(list(las_dir.rglob('*.las'))) + sorted(list(las_dir.rglob('*.laz')))
    
    if not las_files:
        print(f"❌ 严重错误: 在 '{las_dir}' 及其子目录中未找到任何 .las 或 .laz 文件。")
        return None, 0
        
    print(f"找到 {len(las_files)} 个 LAS/LAZ 文件。")

    all_gps_times_list = []
    total_point_count = 0
    
    for path in tqdm(las_files, desc="加载 LAS 文件 (UTC 时间)"):
        try:
            with laspy.open(path) as f:
                las = f.read()
                if not hasattr(las, 'gps_time') or len(las.gps_time) == 0:
                    print(f"  警告: 文件 {path.name} 缺少 'gps_time' 或为空。跳过。")
                    continue
                
                all_gps_times_list.append(las.gps_time)
                total_point_count += len(las.points)
        except Exception as e:
            print(f"  警告: 加载 {path.name} 失败: {e}")
            
    if not all_gps_times_list:
        print("❌ 严重错误: 未能从 LAS 文件中加载任何 GPS 时间数据。")
        return None, 0

    print("正在合并所有 LAS 文件的时间戳...")
    concatenated_gps_times = np.concatenate(all_gps_times_list)
    return concatenated_gps_times, total_point_count


def to_unix_timestamp(dt_obj_local):
    """
    (修改)
    将本地时区的 datetime 对象转换为 Unix 时间戳
    """
    tz = pytz.timezone(LOCAL_TIMEZONE)
    dt_aware = tz.localize(dt_obj_local)
    return dt_aware.timestamp()


def get_flight_status(timestamp, segments):
    """
    (保留)
    根据时间戳判断其所属的航班状态
    """
    for i, seg in enumerate(segments):
        if seg['start'] <= timestamp <= seg['end']:
            return seg['name']
        if i > 0 and segments[i - 1]['end'] < timestamp < seg['start']:
            return f"间隔于 {segments[i - 1]['name'].split(' ')[1]} & {seg['name'].split(' ')[1]}"
    return "任务外"


# --- 3. 主逻辑 ---

try:
    # --- (1/6) (修改) 自动加载航班分段 ---
    print("--- (1/6) 正在从图像元数据加载航班分段 ---")
    flight_segments_dt = load_image_timestamps(image_metadata_dir_path, FLIGHT_GAP_SECONDS)
    
    if not flight_segments_dt:
        print("未能加载航班分段。正在退出。")
        sys.exit(1)

    # --- (2/6) (修改) 转换分段为 Unix ---
    flight_segments_unix = [
        {
            "name": s["name"],
            "start": to_unix_timestamp(s["start_dt"]),
            "end": to_unix_timestamp(s["end_dt"]),
            # (新增) 保存原始 datetime 对象用于打印
            "start_dt_str": s["start_dt"].strftime("%Y-%m-%d %H:%M:%S"),
            "end_dt_str": s["end_dt"].strftime("%Y-%m-%d %H:%M:%S")
        }
        for s in flight_segments_dt
    ]

    print("\n--- (2/6) 航班时间段 (Unix 时间戳) ---")
    total_flight_duration = 0
    for s in flight_segments_unix:
        duration = s['end'] - s['start']
        total_flight_duration += duration
        print(f" {s['name']}: {s['start_dt_str']} -> {s['end_dt_str']} (时长: {duration:.2f} s)")
        
    print(f"总计有效航线时长 (不含间隔): {total_flight_duration:.2f} 秒")
    print("-" * 30)

    # --- (3/6) (修改) 自动加载 LAS 文件 ---
    print("\n--- (3/6) LAS文件处理与对齐 ---")
    gps_times, total_points = load_multiple_las_files(las_dir_path)
    
    if gps_times is None:
        print("未能加载 LAS 数据。正在退出。")
        sys.exit(1)

    min_las_time, max_las_time = np.min(gps_times), np.max(gps_times)
    las_duration = max_las_time - min_las_time
    
    # (!!!) (新增) 打印人类可读的 LAS UTC 时间
    min_las_dt_utc = datetime.datetime.utcfromtimestamp(min_las_time)
    max_las_dt_utc = datetime.datetime.utcfromtimestamp(max_las_time)
    
    print(f"LAS 文件原始 GPS 时间戳 (UTC):")
    print(f"  - 最小值: {min_las_dt_utc} (Raw: {min_las_time})")
    print(f"  - 最大值: {max_las_dt_utc} (Raw: {max_las_time})")
    print(f"  - 总时间跨度: {las_duration:.2f} 秒")
    
    # --- (4/6) (保留) 对齐逻辑 ---
    print("\n--- (4/6) 计算时间偏移量 ---")
    first_flight_start_time = flight_segments_unix[0]["start"]
    time_offset = first_flight_start_time - min_las_time
    print(f"第一个航班开始时间 (本地): {flight_segments_unix[0]['start_dt_str']} ({first_flight_start_time})")
    print(f"第一个 LAS 点时间 (UTC): {min_las_dt_utc} ({min_las_time})")
    print(f"计算出的时间偏移量 (本地 - UTC): {time_offset:.4f} 秒 (约 {time_offset/3600.0:.2f} 小时)")
    
    # (!!!) 检查偏移量是否合理 (例如，对于 UTC+8，应接近 28800)
    if not (28000 < time_offset < 29000): # 假设是 UTC+8
         print(f"⚠️ 警告: 计算出的偏移量 {time_offset/3600.0:.2f} 小时不是 8 小时。")
         print("   请确认您的 LOCAL_TIMEZONE 设置是否正确，以及图像和 LAS 文件是否匹配。")

    adjusted_gps_times = gps_times + time_offset

    # --- (5/6) (保留) 统计与CSV ---
    print("\n--- (5/6) 时间分布统计与CSV生成 ---")
    total_points_in_flights = 0
    for segment in flight_segments_unix:
        points_in_segment = np.sum(
            (adjusted_gps_times >= segment["start"]) & (adjusted_gps_times <= segment["end"])
        )
        total_points_in_flights += points_in_segment
        print(f"在 '{segment['name']}' 时间段内找到 {points_in_segment:,} 个点")

    print(f"\n总点数: {total_points:,}") # (!!!) (修改) 使用新的总数
    print(f"所有航班段内的总点数: {total_points_in_flights:,}")

    # (CSV 生成逻辑 - 保持不变)
    print("正在生成详细CSV日志...")
    start_unix_second = math.floor(np.min(adjusted_gps_times))
    end_unix_second = math.ceil(np.max(adjusted_gps_times))
    bins = np.arange(start_unix_second, end_unix_second + 2)
    point_counts, _ = np.histogram(adjusted_gps_times, bins=bins)

    with open(output_csv_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        log_writer = csv.writer(csvfile)
        header = ["日期时间 (UTC+8)", "校准后_Unix时间戳", "原始_LAS_GPS时间戳", "状态", "该秒内点数"]
        log_writer.writerow(header)
        for i, count in enumerate(point_counts):
            current_unix_ts = start_unix_second + i
            original_gps_ts = current_unix_ts - time_offset
            human_readable_time = datetime.datetime.fromtimestamp(current_unix_ts,
                                                                tz=pytz.timezone(LOCAL_TIMEZONE)).strftime(
                '%Y-%m-%d %H:%M:%S')
            status = get_flight_status(current_unix_ts, flight_segments_unix)
            row = [human_readable_time, current_unix_ts, f"{original_gps_ts:.6f}", status, count]
            log_writer.writerow(row)

    print(f"🎉 成功！详细日志已保存至: {output_csv_file}")

    # --- (6/6) (保留) 可视化 ---
    print("\n--- (6/6) 正在生成时间分布直方图... ---")
    plt.figure(figsize=(15, 7))
    plt.hist(adjusted_gps_times, bins=1000, label="LAS 点数据分布")
    colors = ['green', 'red', 'purple', 'orange', 'blue', 'cyan', 'magenta']
    for i, segment in enumerate(flight_segments_unix):
        color = colors[i % len(colors)]
        plt.axvline(segment['start'], color=color, linestyle='--', label=f"{segment['name']} Start")
        plt.axvline(segment['end'], color=color, linestyle=':', label=f"{segment['name']} End")
    plt.title("LAS 点云时间分布与航班分段对比图")
    plt.xlabel(f"校准后的时间 (Unix Timestamp, {LOCAL_TIMEZONE})")
    plt.ylabel("点数量")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    formatter = plt.FuncFormatter(
        lambda x, pos: datetime.datetime.fromtimestamp(x, tz=pytz.timezone(LOCAL_TIMEZONE)).strftime('%H:%M:%S'))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

except FileNotFoundError as e:
    print(f"❌ 错误: 文件或目录未找到。请检查 'las_dir_path' 和 'image_metadata_dir_path'")
    print(f"  {e}")
except Exception as e:
    print(f"❌ 处理过程中发生意外错误: {e}")
    traceback.print_exc() # 打印详细的错误堆栈
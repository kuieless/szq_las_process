# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm
#
# # --- 解决Matplotlib中文乱码问题 ---
# # 你需要确保你的系统里有这个字体，或者换成你知道的其他中文字体，比如 'SimHei'
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# # -----------------------------------
#
#
# # --- 你需要修改的参数 ---
# # 数据集根目录
# dataset_base_path = r'F:\download\SMBU2\output'
# # 你想处理的文件夹 ('train' 或 'val')
# split_to_process = 'val'
# # 保存可视化结果的输出文件夹
# output_visualization_path = os.path.join(dataset_base_path, f'{split_to_process}_visualized')
# # -------------------------
#
# # 自动创建输出文件夹
# os.makedirs(output_visualization_path, exist_ok=True)
# print(f"可视化结果将保存到: {output_visualization_path}")
#
# # 获取所有要处理的图片列表
# rgbs_dir = os.path.join(dataset_base_path, split_to_process, 'rgbs')
# depth_dir = os.path.join(dataset_base_path, split_to_process, 'depth_dji')
#
# # 获取所有图片文件名（不含后缀）
# image_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(rgbs_dir) if f.endswith('.jpg')])
#
# # 遍历每张图片进行处理
# for image_id in tqdm(image_ids, desc=f'Processing {split_to_process} set'):
#     rgb_image_path = os.path.join(rgbs_dir, f'{image_id}.jpg')
#     depth_map_path = os.path.join(depth_dir, f'{image_id}.npy')
#
#     # 1. 加载RGB和深度图
#     rgb_image = cv2.imread(rgb_image_path)
#     if not os.path.exists(depth_map_path):
#         print(f"警告: 找不到 {image_id}.npy，跳过...")
#         continue
#     depth_map = np.load(depth_map_path)
#
#     # 2. 准备可视化深度图
#     valid_mask = depth_map < 1e5
#     depth_visual = np.zeros(depth_map.shape, dtype=np.uint8)
#     colored_depth_bgr = np.zeros_like(rgb_image)  # 使用BGR格式以匹配cv2
#
#     if np.any(valid_mask):
#         valid_depths = depth_map[valid_mask]
#         min_depth, max_depth = np.min(valid_depths), np.max(valid_depths)
#
#         # 避免除以零的错误
#         if max_depth > min_depth:
#             depth_visual[valid_mask] = ((valid_depths - min_depth) / (max_depth - min_depth)) * 255
#
#         # 应用伪彩色并处理无效区域
#         colored_depth_bgr = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
#         colored_depth_bgr[~valid_mask[:, :, 0]] = 0
#
#         # 3. 将原图和深度可视化图水平拼接
#     # 确保两张图的高度一致
#     h1, w1 = rgb_image.shape[:2]
#     h2, w2 = colored_depth_bgr.shape[:2]
#     if h1 != h2 or w1 != w2:
#         # 如果尺寸不匹配（通常不应该发生），则调整深度图尺寸以匹配RGB图
#         colored_depth_bgr = cv2.resize(colored_depth_bgr, (w1, h1))
#
#     combined_image = np.hstack((rgb_image, colored_depth_bgr))
#
#     # 在图像上添加文字说明
#     cv2.putText(combined_image, f'Original RGB: {image_id}.jpg', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
#                 2)
#     cv2.putText(combined_image, 'Generated Depth Map', (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#
#     # 4. 保存拼接后的图像
#     output_filepath = os.path.join(output_visualization_path, f'{image_id}_comparison.jpg')
#     cv2.imwrite(output_filepath, combined_image)
#
# print("\n批量处理完成！")

import laspy
import numpy as np
import datetime
import pytz
import matplotlib.pyplot as plt
import re

# --- 1. 用户配置 ---
las_file_path = r"F:\download\SMBU2\SMBU1\lidars\terra_las\cloud_merged.las"

# 新的时间基准：从图像中获取的时间规律 (UTC+8)
image_time_segments_raw = [
    {"name": "图像时段 1", "start": "20231116日105324s", "end": "20231116105445"},
    {"name": "图像时段 2", "start": "20231116105703", "end": "20231116105825"},
    {"name": "图像时段 3", "start": "20231116110055", "end": "20231116110127"},
    {"name": "图像时段 4", "start": "20231116110341", "end": "20231116110500"},
]


# --- 2. 辅助函数和数据转换 ---

def parse_custom_time(time_str):
    """解析 YYYYMMDD日HHMMSSs 这种不规则格式"""
    # 使用正则表达式去除所有非数字字符
    digits_only = re.sub(r'\D', '', time_str)
    # 转换为标准格式
    dt = datetime.datetime.strptime(digits_only, "%Y%m%d%H%M%S")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def to_unix_timestamp(dt_str, tz_info):
    """将标准时间字符串转换为 Unix 时间戳"""
    dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    dt_aware = tz_info.localize(dt)
    return dt_aware.timestamp()


# 清理并转换时间格式
image_segments_clean = [
    {"name": s["name"], "start": parse_custom_time(s["start"]), "end": parse_custom_time(s["end"])}
    for s in image_time_segments_raw
]

# 转换为 Unix 时间戳
image_segments_unix = [
    {"name": s["name"], "start": to_unix_timestamp(s["start"], pytz.timezone('Asia/Singapore')),
     "end": to_unix_timestamp(s["end"], pytz.timezone('Asia/Singapore'))}
    for s in image_segments_clean
]

LEAP_SECONDS = 18.0

# --- 3. 主逻辑 ---
try:
    print("--- 基于图像时间进行分析 ---")
    print(f"正在读取 LAS 文件: {las_file_path}")
    with laspy.open(las_file_path) as f:
        las = f.read()
    gps_times = las.gps_time
    min_las_time = np.min(gps_times)

    # 使用第一个图像时段的开始时间作为对齐基准
    first_image_start_time = image_segments_unix[0]["start"]
    time_offset = (first_image_start_time - LEAP_SECONDS) - min_las_time
    adjusted_gps_times = gps_times + time_offset

    print(f"LAS 文件原始 GPS 时间起点: {min_las_time}")
    print(f"图像报告 Unix 时间起点: {first_image_start_time}")
    print(f"已加入 {LEAP_SECONDS} 秒闰秒修正")
    print(f"计算出的新时间偏移量: {time_offset:.4f} 秒")

    # --- 4. 时长与间隔对比分析 ---
    print("\n--- 时长与间隔对比分析 (图像时间 vs LAS数据块) ---")

    print("【图像时段数据】:")
    image_durations = []
    for i, segment in enumerate(image_segments_unix):
        duration = segment['end'] - segment['start']
        image_durations.append(duration)
        print(f"  {segment['name']} 持续时长: {duration:.2f} 秒")
        if i > 0:
            gap = segment['start'] - image_segments_unix[i - 1]['end']
            print(f"  (间隔于时段 {i}): {gap:.2f} 秒")

    # 从LAS数据中识别数据块
    counts, bins = np.histogram(adjusted_gps_times, bins=1000)
    bin_width = bins[1] - bins[0]
    threshold = counts.mean() * 0.1
    in_data_block = counts > threshold
    block_edges = np.diff(in_data_block.astype(int))
    start_indices = np.where(block_edges == 1)[0]
    end_indices = np.where(block_edges == -1)[0]

    if len(start_indices) > 0 and len(end_indices) > 0:
        if end_indices[0] < start_indices[0]: end_indices = end_indices[1:]
        if start_indices[-1] > end_indices[-1]: start_indices = start_indices[:-1]
        min_len = min(len(start_indices), len(end_indices))

        print("\n【LAS 数据块分析】:")
        for i in range(min_len):
            start_time = bins[start_indices[i]]
            end_time = bins[end_indices[i]]
            duration = end_time - start_time
            print(f"  数据块 {i + 1} 持续时长: {duration:.2f} 秒")
            if i > 0:
                prev_end_time = bins[end_indices[i - 1]]
                gap = start_time - prev_end_time
                print(f"  (间隔于数据块 {i}): {gap:.2f} 秒")
    else:
        print("\n无法在LAS数据中清晰地识别出数据块。")

    # --- 5. 可视化 ---
    print("\n正在生成新的对比直方图...")
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.hist(adjusted_gps_times, bins=1000, label="LAS 点数据分布", zorder=2)
    colors = ['cyan', 'magenta', 'yellow', 'lime']
    for i, segment in enumerate(image_segments_unix):
        color = colors[i % len(colors)]
        ax.axvspan(segment['start'], segment['end'], color=color, alpha=0.4, label=f"{segment['name']}", zorder=1)
    ax.set_title("LAS点云时间分布 vs 图像拍摄时间窗口", fontsize=16)
    ax.set_xlabel("校准后的时间 (Unix Timestamp)", fontsize=12)
    ax.set_ylabel("点数量", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
    formatter = plt.FuncFormatter(
        lambda x, pos: datetime.datetime.fromtimestamp(x, tz=pytz.timezone('Asia/Singapore')).strftime('%H:%M:%S'))
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"处理过程中发生错误: {e}")
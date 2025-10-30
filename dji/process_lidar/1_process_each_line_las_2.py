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
# import math
#
# # --- 项目路径设置 ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# grandparent_dir = os.path.dirname(parent_dir)
# if grandparent_dir not in sys.path:
#     sys.path.append(grandparent_dir)
# from dji.opt import _get_opts
#
#
# def main(hparams):
#     # ==============================================================================
#     # --- 用户配置区域 ---
#     # 定义围绕照片时间戳的总窗口半宽度 (秒)
#     # 例如 10.0 表示向前看10秒，向后看10秒，总共 20 秒
#     TIME_WINDOW_HALF_WIDTH = 30.0
#     # ==============================================================================
#
#     dataset_path = Path(hparams.dataset_path)
#     las_output_path = Path(hparams.las_output_path)
#     chunks_output_dir = las_output_path / 'lidar_chunks'
#     chunks_output_dir.mkdir(parents=True, exist_ok=True)
#     print(f"点云块文件将保存到: {chunks_output_dir}")
#
#     # --- 1. 加载LiDAR点云数据 ---
#     print("正在加载LAS文件...")
#     try:
#         in_file = laspy.file.File(hparams.las_path, mode='r')
#         with in_file:
#             x, y, z = in_file.x, in_file.y, in_file.z
#             point_format = in_file.point_format
#             has_color = 'red' in point_format.lookup and 'green' in point_format.lookup and 'blue' in point_format.lookup
#             if has_color:
#                 r = (in_file.red / 65536.0 * 255).astype(np.uint8)
#                 g = (in_file.green / 65536.0 * 255).astype(np.uint8)
#                 b = (in_file.blue / 65536.0 * 255).astype(np.uint8)
#             else:
#                 print("警告: LAS文件中未找到颜色信息，将使用默认白色。")
#                 r = np.full(len(x), 255, dtype=np.uint8);
#                 g = r;
#                 b = r
#             lidars = np.array(list(zip(x, y, z, r, g, b)))
#             if np.any(np.isnan(in_file.gps_time)) or np.all(in_file.gps_time == in_file.gps_time[0]):
#                 raise ValueError("LAS GPS 时间戳无效")
#             diff_gps_time = np.array(in_file.gps_time - in_file.gps_time[0])
#             lidar_start_relative_time = diff_gps_time.min()
#             lidar_end_relative_time = diff_gps_time.max()
#     except Exception as e:
#         print(f"❌ 错误：加载LAS文件失败: {e}")
#         sys.exit(1)
#     print(f"LAS加载完成。相对时间范围: [{lidar_start_relative_time:.3f}s, {lidar_end_relative_time:.3f}s]")
#
#     # --- 2. (核心修改) 查找【所有实际存在】的元数据文件 ---
#     print("\n正在查找所有存在的 image_metadata 文件...")
#     train_paths = sorted(list((dataset_path / 'train' / 'image_metadata').glob('*.pt')))
#     val_paths = sorted(list((dataset_path / 'val' / 'image_metadata').glob('*.pt')))
#     all_existing_metadata_paths = train_paths + val_paths
#     print(f"共找到 {len(all_existing_metadata_paths)} 个存在的 .pt 文件。")
#
#     if not all_existing_metadata_paths:
#         sys.exit("❌ 错误：在 train/ 或 val/image_metadata 目录下未找到任何 .pt 文件。")
#
#     # --- 3. 加载找到的文件并按EXIF时间排序 ---
#     print("\n加载并排序找到的元数据文件...")
#     all_metadata = []
#     successful_loads = 0
#     failed_loads = 0
#     print("开始加载元数据文件:")
#     for idx, path in enumerate(all_existing_metadata_paths):
#         print(f"  [{(idx + 1):>3d}/{len(all_existing_metadata_paths)}] 尝试加载: {path.name} ... ", end="")
#         try:
#             metadata = torch.load(path, weights_only=False)
#             if 'meta_tags' in metadata and 'EXIF DateTimeOriginal' in metadata['meta_tags']:
#                 # (新增) 保存原始文件索引 (文件名去掉 .pt)
#                 original_index = int(path.stem)
#                 all_metadata.append({'data': metadata, 'path': path, 'original_index': original_index})
#                 print("成功!")
#                 successful_loads += 1
#             else:
#                 print("失败 (缺少键 'meta_tags' 或 'EXIF DateTimeOriginal')")
#                 failed_loads += 1
#         except Exception as e:
#             print(f"失败 (读取错误: {e})")
#             failed_loads += 1
#
#     print(f"\n元数据加载完成。成功: {successful_loads} 个, 失败/跳过: {failed_loads} 个。")
#
#     if not all_metadata: sys.exit("❌ 错误：未能成功加载任何有效的元数据文件。")
#
#     # 按时间排序加载成功的文件
#     try:
#         all_metadata.sort(key=lambda x: x['data']['meta_tags']['EXIF DateTimeOriginal'].values)
#         print("照片元数据排序完成。")
#     except Exception as e:
#         print(f"❌ 错误：对加载的元数据进行排序时失败: {e}")
#         sys.exit(1)
#
#     # --- 4. 计算相对时间戳 (基于排序后的列表) ---
#     print("\n计算照片相对时间...")
#     sorted_photo_relative_times = []
#     origin_timestamp_unix = None
#     first_photo_dt_local = datetime.datetime.strptime(
#         all_metadata[0]['data']['meta_tags']['EXIF DateTimeOriginal'].values, '%Y:%m:%d %H:%M:%S')
#     origin_timestamp_unix = first_photo_dt_local.timestamp()
#
#     # 为排序后的列表中的每个元数据计算相对时间
#     for item in all_metadata:
#         time_str = item['data']['meta_tags']['EXIF DateTimeOriginal'].values
#         dt_local = datetime.datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
#         time_difference = dt_local.timestamp() - origin_timestamp_unix
#         # 将相对时间存回字典，方便后续使用
#         item['relative_time'] = time_difference
#
#     print(f"已为 {len(all_metadata)} 张有效照片计算相对时间。")
#     print(f"照片相对时间范围: [{all_metadata[0]['relative_time']:.3f}s, {all_metadata[-1]['relative_time']:.3f}s]")
#
#     # --- 5. (核心修改) 为【每一个】加载成功的照片生成点云块 ---
#     print("\n开始为每一张加载成功的照片生成点云块...")
#
#     # 不再使用 TARGET_PHOTO_INDICES，直接遍历 all_metadata
#     for item in tqdm(all_metadata, desc="生成点云块"):
#
#         target_index = item['original_index']  # 使用原始文件索引
#         photo_time = item['relative_time']  # 使用计算好的相对时间
#
#         print(f"\n--- 处理照片索引: {target_index} (相对时间: {photo_time:.3f}s) ---", flush=True)  # 添加 flush
#
#         photo_chunks = []
#         chunk_time_offsets = []
#
#         start_second_offset = math.floor(-TIME_WINDOW_HALF_WIDTH)
#         end_second_offset = math.ceil(TIME_WINDOW_HALF_WIDTH)
#
#         # (修改) 移除内层 tqdm，避免嵌套进度条混乱
#         for sec_offset in range(start_second_offset, end_second_offset):
#             chunk_start_time = photo_time + sec_offset
#             chunk_end_time = chunk_start_time + 1.0
#             chunk_start_time = max(chunk_start_time, lidar_start_relative_time)
#             chunk_end_time = min(chunk_end_time, lidar_end_relative_time)
#             if chunk_start_time >= chunk_end_time: continue
#
#             mask = (diff_gps_time >= chunk_start_time) & (diff_gps_time < chunk_end_time)
#             points_in_chunk = lidars[mask]
#             photo_chunks.append(points_in_chunk)
#             chunk_time_offsets.append(sec_offset)
#
#         output_data = {
#             "target_photo_index": target_index,
#             "relative_photo_time": photo_time,
#             "time_window_half_width": TIME_WINDOW_HALF_WIDTH,
#             "chunk_time_offsets": chunk_time_offsets,
#             "point_cloud_chunks": photo_chunks
#         }
#         # 使用原始索引命名文件
#         output_filename = f'points_lidar_chunks_{target_index:06d}.pkl'
#         output_filepath = chunks_output_dir / output_filename
#         try:
#             with open(output_filepath, "wb") as file:
#                 pickle.dump(output_data, file)
#             print(f"  已为照片 {target_index} 保存 {len(photo_chunks)} 个点云块到: {output_filename}", flush=True)
#         except Exception as e:
#             print(f"  ❌ 错误: 保存文件 {output_filename} 失败: {e}", flush=True)
#
#     print("\n" + "=" * 25 + " 点云块生成完毕 " + "=" * 25)
#
#
# if __name__ == '__main__':
#     main(_get_opts())


import numpy as np
from tqdm import tqdm
import laspy
import torch
import datetime
from pathlib import Path
import pickle
import os
import sys
import math

# --- 项目路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)
from dji.opt import _get_opts


def main(hparams):
    # ==============================================================================
    # --- 用户配置区域 ---
    # 1. 定义围绕照片时间戳的总窗口半宽度 (秒)
    #    例如 30.0 表示向前看30秒，向后看30秒，总共 60 秒的范围
    TIME_WINDOW_HALF_WIDTH = 80.0

    # 2. (新增) 定义每个点云块的持续时间 (秒)
    #    例如 3.0 表示每个块包含3秒的数据
    CHUNK_DURATION_SECONDS = 1.0

    # 3. (新增) 定义创建块的步长 (秒)
    #    - 如果 STEP 等于 DURATION，块将是连续且不重叠的 (例如 [0,3], [3,6], ...)
    #    - 如果 STEP 小于 DURATION，块将是重叠的 (例如 DURATION=3, STEP=1 -> [0,3], [1,4], [2,5], ...)
    CHUNK_STEP_SECONDS = 1.0
    # ==============================================================================

    dataset_path = Path(hparams.dataset_path)
    las_output_path = Path(hparams.las_output_path)
    chunks_output_dir = las_output_path / 'lidar_chunks'
    chunks_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"点云块文件将保存到: {chunks_output_dir}")
    print(f"配置: 窗口半宽={TIME_WINDOW_HALF_WIDTH}s, 块时长={CHUNK_DURATION_SECONDS}s, 步长={CHUNK_STEP_SECONDS}s")


    # --- 1. 加载LiDAR点云数据 ---
    print("\n正在加载LAS文件...")
    try:
        in_file = laspy.file.File(hparams.las_path, mode='r')
        with in_file:
            x, y, z = in_file.x, in_file.y, in_file.z
            point_format = in_file.point_format
            has_color = 'red' in point_format.lookup and 'green' in point_format.lookup and 'blue' in point_format.lookup
            if has_color:
                r = (in_file.red / 65536.0 * 255).astype(np.uint8)
                g = (in_file.green / 65536.0 * 255).astype(np.uint8)
                b = (in_file.blue / 65536.0 * 255).astype(np.uint8)
            else:
                print("警告: LAS文件中未找到颜色信息，将使用默认白色。")
                r = np.full(len(x), 255, dtype=np.uint8);
                g = r;
                b = r
            lidars = np.array(list(zip(x, y, z, r, g, b)))
            if np.any(np.isnan(in_file.gps_time)) or np.all(in_file.gps_time == in_file.gps_time[0]):
                raise ValueError("LAS GPS 时间戳无效")
            diff_gps_time = np.array(in_file.gps_time - in_file.gps_time[0])
            lidar_start_relative_time = diff_gps_time.min()
            lidar_end_relative_time = diff_gps_time.max()
    except Exception as e:
        print(f"❌ 错误：加载LAS文件失败: {e}")
        sys.exit(1)
    print(f"LAS加载完成。相对时间范围: [{lidar_start_relative_time:.3f}s, {lidar_end_relative_time:.3f}s]")

    # --- 2. 查找【所有实际存在】的元数据文件 ---
    print("\n正在查找所有存在的 image_metadata 文件...")
    train_paths = sorted(list((dataset_path / 'train' / 'image_metadata').glob('*.pt')))
    val_paths = sorted(list((dataset_path / 'val' / 'image_metadata').glob('*.pt')))
    all_existing_metadata_paths = train_paths + val_paths
    print(f"共找到 {len(all_existing_metadata_paths)} 个存在的 .pt 文件。")

    if not all_existing_metadata_paths:
        sys.exit("❌ 错误：在 train/ 或 val/image_metadata 目录下未找到任何 .pt 文件。")

    # --- 3. 加载找到的文件并按EXIF时间排序 ---
    print("\n加载并排序找到的元数据文件...")
    all_metadata = []
    # ... (此部分代码保持不变) ...
    successful_loads = 0
    failed_loads = 0
    print("开始加载元数据文件:")
    for idx, path in enumerate(all_existing_metadata_paths):
        print(f"  [{(idx + 1):>3d}/{len(all_existing_metadata_paths)}] 尝试加载: {path.name} ... ", end="")
        try:
            metadata = torch.load(path, weights_only=False)
            if 'meta_tags' in metadata and 'EXIF DateTimeOriginal' in metadata['meta_tags']:
                original_index = int(path.stem)
                all_metadata.append({'data': metadata, 'path': path, 'original_index': original_index})
                print("成功!")
                successful_loads += 1
            else:
                print("失败 (缺少键 'meta_tags' 或 'EXIF DateTimeOriginal')")
                failed_loads += 1
        except Exception as e:
            print(f"失败 (读取错误: {e})")
            failed_loads += 1
    print(f"\n元数据加载完成。成功: {successful_loads} 个, 失败/跳过: {failed_loads} 个。")
    if not all_metadata: sys.exit("❌ 错误：未能成功加载任何有效的元数据文件。")
    try:
        all_metadata.sort(key=lambda x: x['data']['meta_tags']['EXIF DateTimeOriginal'].values)
        print("照片元数据排序完成。")
    except Exception as e:
        print(f"❌ 错误：对加载的元数据进行排序时失败: {e}")
        sys.exit(1)


    # --- 4. 计算相对时间戳 (基于排序后的列表) ---
    print("\n计算照片相对时间...")
    # ... (此部分代码保持不变) ...
    origin_timestamp_unix = None
    first_photo_dt_local = datetime.datetime.strptime(
        all_metadata[0]['data']['meta_tags']['EXIF DateTimeOriginal'].values, '%Y:%m:%d %H:%M:%S')
    origin_timestamp_unix = first_photo_dt_local.timestamp()
    for item in all_metadata:
        time_str = item['data']['meta_tags']['EXIF DateTimeOriginal'].values
        dt_local = datetime.datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
        time_difference = dt_local.timestamp() - origin_timestamp_unix
        item['relative_time'] = time_difference
    print(f"已为 {len(all_metadata)} 张有效照片计算相对时间。")
    print(f"照片相对时间范围: [{all_metadata[0]['relative_time']:.3f}s, {all_metadata[-1]['relative_time']:.3f}s]")


    # --- 5. 为【每一个】加载成功的照片生成点云块 ---
    print("\n开始为每一张加载成功的照片生成点云块...")

    for item in tqdm(all_metadata, desc="生成点云块"):
        target_index = item['original_index']
        photo_time = item['relative_time']

        print(f"\n--- 处理照片索引: {target_index} (相对时间: {photo_time:.3f}s) ---", flush=True)

        photo_chunks = []
        chunk_time_offsets = []

        # 定义整个时间窗口的开始和结束偏移
        window_start_offset = -TIME_WINDOW_HALF_WIDTH
        window_end_offset = TIME_WINDOW_HALF_WIDTH

        # --- (核心修改) 使用 while 循环和新参数来创建块 ---
        # 从窗口的起始点开始
        current_start_offset = window_start_offset
        while current_start_offset + CHUNK_DURATION_SECONDS <= window_end_offset:
            # 计算当前块的绝对起止时间
            chunk_start_time = photo_time + current_start_offset
            chunk_end_time = chunk_start_time + CHUNK_DURATION_SECONDS

            # 确保块的时间范围不超出整个LiDAR数据的时间范围
            # (这一步逻辑保持不变，但很重要)
            effective_start = max(chunk_start_time, lidar_start_relative_time)
            effective_end = min(chunk_end_time, lidar_end_relative_time)

            # 如果有效时间范围小于0，则跳过
            if effective_start >= effective_end:
                # 移动到下一个步长位置
                current_start_offset += CHUNK_STEP_SECONDS
                continue

            # 筛选出在当前时间块内的点
            mask = (diff_gps_time >= effective_start) & (diff_gps_time < effective_end)
            points_in_chunk = lidars[mask]

            # 存储块和对应的开始时间偏移量 (相对于照片时间)
            photo_chunks.append(points_in_chunk)
            # 注意：我们将偏移量保存为整数，以保持与下一个脚本的文件名格式兼容
            chunk_time_offsets.append(int(round(current_start_offset)))

            # 移动到下一个步长位置
            current_start_offset += CHUNK_STEP_SECONDS
        # --- 修改结束 ---

        output_data = {
            "target_photo_index": target_index,
            "relative_photo_time": photo_time,
            "time_window_half_width": TIME_WINDOW_HALF_WIDTH,
            "chunk_time_offsets": chunk_time_offsets,
            "point_cloud_chunks": photo_chunks
        }
        output_filename = f'points_lidar_chunks_{target_index:06d}.pkl'
        output_filepath = chunks_output_dir / output_filename
        try:
            with open(output_filepath, "wb") as file:
                pickle.dump(output_data, file)
            print(f"  已为照片 {target_index} 保存 {len(photo_chunks)} 个点云块到: {output_filename}", flush=True)
        except Exception as e:
            print(f"  ❌ 错误: 保存文件 {output_filename} 失败: {e}", flush=True)

    print("\n" + "=" * 25 + " 点云块生成完毕 " + "=" * 25)


if __name__ == '__main__':
    main(_get_opts())

#    python 1_process_each_line_las_2.py --dataset_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output --las_path F:\download\SMBU2\SMBU1\lidars\terra_las\cloud_merged.las --las_output_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output --num_val 1

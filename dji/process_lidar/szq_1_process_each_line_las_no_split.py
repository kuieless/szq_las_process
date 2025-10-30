# # # # # # 脚本功能：
# # # # # # 1. (已修改) 加载并按EXIF时间排序照片元数据 (移除了 train/val 划分)。
# # # # # # 2. (已修改) 支持加载一个目录下的多个 .las/.laz 文件块，并将其合并。
# # # # # # 3. (已修改) (核心修改) 提供两种点云划分策略 ('exposure_time', 'fixed_window')。
# # # # # # 4. (新增) (GPU加速) 将合并后的点云和时间戳上传到GPU，使用PyTorch进行高速时间窗口切片。
# # # # # # 5. 输出分段后的点云列表 (points_lidar_list.pkl)。

# # # # # from plyfile import PlyData
# # # # # import numpy as np
# # # # # from tqdm import tqdm

# # # # # import laspy
# # # # # import torch
# # # # # import datetime
# # # # # from pathlib import Path
# # # # # import pickle
# # # # # import os
# # # # # import sys

# # # # # # --- 项目路径设置 (与您的脚本保持一致) ---
# # # # # current_dir = os.path.dirname(os.path.abspath(__file__))
# # # # # parent_dir = os.path.dirname(current_dir)
# # # # # grandparent_dir = os.path.dirname(parent_dir)
# # # # # if grandparent_dir not in sys.path:
# # # # #     sys.path.append(grandparent_dir)
# # # # # from dji.opt import _get_opts


# # # # # # --- 新增辅助函数，用于解析曝光时间字符串 ---
# # # # # def parse_exposure_time(value):
# # # # #     """(已修正) 将 ['1/2000'] 或 '1/2000' 这样的值转换为秒(float)"""
# # # # #     # 检查输入是否为列表，如果是，则提取第一个元素
# # # # #     if isinstance(value, list) and len(value) > 0:
# # # # #         time_str = value[0]
# # # # #     elif isinstance(value, str):
# # # # #         time_str = value
# # # # #     else:
# # # # #         # 如果输入既不是列表也不是字符串，无法处理
# # # # #         print(f"警告: 曝光时间格式未知 '{value}', 将使用默认值 0.001 秒。")
# # # # #         return 0.001

# # # # #     try:
# # # # #         if '/' in time_str:
# # # # #             num, den = map(float, time_str.split('/'))
# # # # #             return num / den
# # # # #         else:
# # # # #             return float(time_str)
# # # # #     except (ValueError, ZeroDivisionError, TypeError):
# # # # #         # 增加了对TypeError的捕获，以防万一
# # # # #         print(f"警告: 无法解析曝光时间字符串 '{time_str}', 将使用默认值 0.001 秒。")
# # # # #         return 0.001


# # # # # def main(hparams):
# # # # #     # ==============================================================================
# # # # #     # --- 新增配置：在这里选择您的点云划分方法 ---
# # # # #     #
# # # # #     # 'exposure_time' -> 使用每张照片的曝光时间 (最严谨，用于生成GT)。
# # # # #     # 'fixed_window'  -> 使用下方定义的固定时间窗口 (点云更密集，用于场景理解)。
# # # # #     SLICING_METHOD = 'fixed_window'

# # # # #     # 当 SLICING_METHOD 设置为 'fixed_window' 时，此参数生效
# # # # #     # 3.0 代表以照片时间戳为中心，前后各取1.5秒，总共3秒的窗口。
# # # # #     FIXED_WINDOW_SECONDS = 14.0
# # # # #     # ==============================================================================

# # # # #     dataset_path = Path(hparams.dataset_path)
# # # # #     las_output_path = Path(hparams.las_output_path)
# # # # #     las_output_path.mkdir(parents=True, exist_ok=True)

# # # # #     # --- 1. (已修改) 加载多个LiDAR点云数据块 ---
# # # # #     print("正在扫描并加载 LAS/LAZ 文件...")
# # # # #     # (修改) hparams.las_path 现应为 hparams.las_dir
# # # # #     # las_dir = Path(hparams.las_dir) 
# # # # # # (修改) 我们复用 --las_path 参数，但将其视为一个目录
# # # # #     las_dir = Path(hparams.las_path)
# # # # #     las_files = sorted(list(las_dir.glob('*.las'))) + sorted(list(las_dir.glob('*.laz')))
    
# # # # #     if not las_files:
# # # # #         print(f"❌ 严重错误: 在 '{las_dir}' 中未找到任何 .las 或 .laz 文件。")
# # # # #         sys.exit(1)
    
# # # # #     print(f"找到 {len(las_files)} 个 LAS/LAZ 文件。")

# # # # #     all_lidars = []
# # # # #     all_gps_times = []
    
# # # # #     # for las_file_path in tqdm(las_files, desc="Loading LAS/LAZ files"):
# # # # #     #     try:
# # # # #     #         with laspy.open(las_file_path) as in_file:
# # # # #     #             x, y, z = in_file.x, in_file.y, in_file.z
# # # # #     #             r = (in_file.red / 65536.0 * 255).astype(np.uint8)
# # # # #     #             g = (in_file.green / 65536.0 * 255).astype(np.uint8)
# # # # #     #             b = (in_file.blue / 65536.0 * 255).astype(np.uint8)
                
# # # # #     #             all_lidars.append(np.array(list(zip(x, y, z, r, g, b))))
# # # # #     #             all_gps_times.append(np.array(in_file.gps_time))
# # # # #     #     except Exception as e:
# # # # #     #         print(f"警告: 加载文件 {las_file_path} 失败: {e}")
# # # # #     for las_file_path in tqdm(las_files, desc="Loading LAS/LAZ files"):
# # # # #         try:
# # # # #             with laspy.open(las_file_path) as in_file:
                
# # # # #                 # (!!!) 关键修复 (!!!)
# # # # #                 # 必须调用 .read() 来获取 LasData 对象
# # # # #                 las = in_file.read()
                
# # # # #                 # 现在从 'las' (LasData) 而不是 'in_file' (LasReader) 读取
# # # # #                 x, y, z = las.x, las.y, las.z
                
# # # # #                 # 颜色和时间戳也从 'las' 读取
# # # # #                 r = (las.red / 65536.0 * 255).astype(np.uint8)
# # # # #                 g = (las.green / 65536.0 * 255).astype(np.uint8)
# # # # #                 b = (las.blue / 65536.0 * 255).astype(np.uint8)
                
# # # # #                 all_lidars.append(np.array(list(zip(x, y, z, r, g, b))))
# # # # #                 all_gps_times.append(np.array(las.gps_time))
                
# # # # #         except Exception as e:
# # # # #             print(f"警告: 加载文件 {las_file_path} 失败: {e}")

# # # # #     print("正在合并所有点云块...")
# # # # #     lidars = np.concatenate(all_lidars, axis=0)
# # # # #     gps_times = np.concatenate(all_gps_times, axis=0)
    
# # # # #     # (新增) 必须按时间排序，以确保切片正确
# # # # #     print(f"正在按 GPS 时间排序 {len(lidars)} 个点...")
# # # # #     sort_indices = np.argsort(gps_times)
# # # # #     lidars = lidars[sort_indices]
# # # # #     gps_times = gps_times[sort_indices]
    
# # # # #     diff_gps_time = gps_times - gps_times[0]
# # # # #     print("LAS文件加载、合并、排序并计算相对时间戳完成。")


# # # # #     # --- (新增) GPU 加速设置 ---
# # # # #     device = None
# # # # #     if torch.cuda.is_available():
# # # # #         try:
# # # # #             device = torch.device("cuda:0")
# # # # #             print(f"\n--- 检测到 CUDA！正在尝试将 {len(lidars)} 个点移动到 GPU... ---")
# # # # #             # 将数据上传到 GPU
# # # # #             diff_gps_time_gpu = torch.tensor(diff_gps_time, device=device)
# # # # #             lidars_gpu = torch.tensor(lidars, device=device)
# # # # #             print("--- 数据成功移动到 GPU。将使用 GPU 进行切片。 ---")
# # # # #         except Exception as e:
# # # # #             print(f"--- GPU 错误 (可能内存不足 OOM): {e} ---")
# # # # #             print("--- 将回退到 CPU (速度较慢)。 ---")
# # # # #             device = torch.device("cpu")
# # # # #             diff_gps_time_gpu = torch.tensor(diff_gps_time) # 存在 CPU
# # # # #             lidars_gpu = torch.tensor(lidars)         # 存在 CPU
# # # # #     else:
# # # # #         print("\n--- 未检测到 CUDA。将使用 CPU。 ---")
# # # # #         device = torch.device("cpu")
# # # # #         diff_gps_time_gpu = torch.tensor(diff_gps_time)
# # # # #         lidars_gpu = torch.tensor(lidars)

# # # # #     # 释放CPU内存 (如果已成功上传到GPU)
# # # # #     if device.type == 'cuda':
# # # # #         del lidars
# # # # #         del gps_times
# # # # #         del diff_gps_time
# # # # #         import gc
# # # # #         gc.collect()
# # # # #         print("--- 已释放 CPU 上的点云内存。 ---")


# # # # #     # --- 2. (已修改) 加载并按EXIF时间排序照片元数据 ---
# # # # #     # (修改) 移除 train/val 划分
# # # # #     image_metadata_dir = dataset_path / 'image_metadata'
# # # # #     if not image_metadata_dir.exists():
# # # # #         print(f"❌ 严重错误: 找不到元数据目录 '{image_metadata_dir}'")
# # # # #         print("请先运行第0个脚本。")
# # # # #         sys.exit(1)
        
# # # # #     all_metadata_paths = sorted(list(image_metadata_dir.iterdir()))

# # # # #     print("\n正在加载所有图像元数据并按EXIF拍摄时间排序...")
# # # # #     all_metadata = []
# # # # #     for path in tqdm(all_metadata_paths, desc="Loading metadata"):
# # # # #         # weights_only=False 是为了加载包含EXIF的dict (非安全)
# # # # #         # 切换到 torch.load(path, map_location='cpu') 更安全，如果元数据是张量
# # # # #         # 但看起来元数据是复杂的dict，所以 weights_only=False 是必须的
# # # # #         metadata = torch.load(path, weights_only=False) 
# # # # #         all_metadata.append({'data': metadata, 'path': path})

# # # # #     all_metadata.sort(key=lambda x: x['data']['meta_tags']['EXIF DateTimeOriginal'].values)
# # # # #     print("照片元数据排序完成。")

# # # # #     # --- 3. 提取精确时间戳和曝光时间 ---
# # # # #     print("\n正在提取每张照片的精确时间和曝光时间...")
# # # # #     sorted_photo_info = []
# # # # #     origin_timestamp = None

# # # # #     for item in tqdm(all_metadata, desc="Extracting photo info"):
# # # # #         metadata = item['data']
# # # # #         time_str = metadata['meta_tags']['EXIF DateTimeOriginal'].values
# # # # #         dt_local = datetime.datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
# # # # #         unix_timestamp = dt_local.timestamp()

# # # # #         if origin_timestamp is None:
# # # # #             origin_timestamp = unix_timestamp
# # # # #         time_difference = unix_timestamp - origin_timestamp

# # # # #         exposure_time_key = 'EXIF ExposureTime'
# # # # #         if exposure_time_key in metadata['meta_tags']:
# # # # #             # 这里我们直接传递.values的原始输出（可能是列表）
# # # # #             exposure_time_value = metadata['meta_tags'][exposure_time_key].values
# # # # #             exposure_time_sec = parse_exposure_time(exposure_time_value)
# # # # #         else:
# # # # #             exposure_time_sec = 0.001

# # # # #         sorted_photo_info.append({
# # # # #             "relative_time": time_difference,
# # # # #             "exposure_time": exposure_time_sec
# # # # #         })

# # # # #     # --- 4. (核心修改 & GPU加速) 根据选择的策略进行点云划分 ---
# # # # #     print(f"\n当前划分策略: '{SLICING_METHOD}'")
# # # # #     if SLICING_METHOD == 'fixed_window':
# # # # #         print(f"固定窗口大小: {FIXED_WINDOW_SECONDS} 秒")

# # # # #     points_lidar_list = []
# # # # #     total_points_matched = 0
# # # # #     non_empty_segments = 0

# # # # #     for info in tqdm(sorted_photo_info, desc=f"Slicing point cloud (on {device.type})"):
# # # # #         photo_time = info['relative_time']

# # # # #         # 根据选择的策略，决定窗口的宽度
# # # # #         if SLICING_METHOD == 'fixed_window':
# # # # #             window_width = FIXED_WINDOW_SECONDS
# # # # #         elif SLICING_METHOD == 'exposure_time':
# # # # #             window_width = info['exposure_time']
# # # # #         else:
# # # # #             raise ValueError("SLICING_METHOD 必须是 'exposure_time' 或 'fixed_window'")

# # # # #         # 计算时间窗口
# # # # #         start_window = photo_time - (window_width / 2.0)
# # # # #         end_window = photo_time + (window_width / 2.0)

# # # # #         # --- (!!!) GPU 加速切片 (!!!) ---
# # # # #         # 1. 在 GPU 上计算布尔掩码
# # # # #         bool_mask = (diff_gps_time_gpu >= start_window) & (diff_gps_time_gpu < end_window)
        
# # # # #         # 2. 在 GPU 上索引
# # # # #         points_in_segment_gpu = lidars_gpu[bool_mask]

# # # # #         # 3. (关键) 将结果移回 CPU 以便保存到列表和 pickle
# # # # #         points_in_segment_cpu = points_in_segment_gpu.cpu().numpy()
# # # # #         # --- (!!!) 加速结束 (!!!) ---

# # # # #         # 更新统计
# # # # #         num_points = len(points_in_segment_cpu)
# # # # #         if num_points > 0:
# # # # #             non_empty_segments += 1
# # # # #             total_points_matched += num_points
# # # # #         points_lidar_list.append(points_in_segment_cpu)

# # # # #     # --- 5. 保存结果并生成总结报告 ---
# # # # #     print("\n划分完成，正在保存结果...")
# # # # #     output_pkl_path = las_output_path / 'points_lidar_list.pkl'
# # # # #     with open(str(output_pkl_path), "wb") as file:
# # # # #         pickle.dump(points_lidar_list, file)
# # # # #     print(f"点云分段列表已保存至: {str(output_pkl_path)}")

# # # # #     total_segments = len(points_lidar_list)
# # # # #     print("\n" + "=" * 25 + " 最终匹配结果总结 " + "=" * 25)
# # # # #     print(f"总共为 {total_segments} 张照片生成了时间分段。")
# # # # #     print(f"其中有 {non_empty_segments} 个分段成功匹配到了点云 (非空)。")
# # # # #     print(f"总共匹配到的点云数量为: {total_points_matched}")
# # # # #     if total_segments > 0:
# # # # #         avg_points = total_points_matched / non_empty_segments if non_empty_segments > 0 else 0
# # # # #         print(f"平均每个非空分段包含 {avg_points:.1f} 个点。")

# # # # #     if non_empty_segments > total_segments * 0.8 and total_points_matched > 1000:
# # # # #         print("✅ 匹配情况良好！大部分时间段都找到了对应的点云。")
# # # # #     elif non_empty_segments > 0:
# # # # #         print("⚠️ 注意：部分时间段未能匹配到点云，请检查照片和LiDAR数据的时间覆盖范围是否完全重合。")
# # # # #     else:
# # # # #         print("❌ 严重错误：所有时间段都未能匹配到点云！请重新检查时间戳转换逻辑或数据源。")
# # # # #     print("=" * 70)
# # # # #     print('脚本执行完毕。')


# # # # # if __name__ == '__main__':
# # # # #     main(_get_opts())

# # # # # # (修改后的示例命令)
# # # # # # SBMU     python 1_process_multi_las_gpu.py --dataset_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output_SBMU --las_dir F:\download\SMBU2\SMBU1\lidars\terra_las --las_output_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output_SBMU

# # # # # # SZIIT    python 1_process_multi_las_gpu.py --dataset_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output_SZTTI --las_dir H:\BaiduNetdiskDownload\LiDAR_Raw\SZIIT\SZTIH018\lidars\terra_las --las_output_path F:\download\SMBU2\Aerial_lifting_early\dji\process_lidar\output_SZTTI


# # # # # 脚本功能：
# # # # # 1. (已修改) 加载并按EXIF时间排序照片元数据 (移除了 train/val 划分)。
# # # # # 2. (已修改) 支持加载一个目录下的多个 .las/.laz 文件块，并将其合并。
# # # # # 3. (已修改) (核心修改) 提供两种点云划分策略 ('exposure_time', 'fixed_window')。
# # # # # 4. (新增) (GPU加速) 将合并后的点云和时间戳上传到GPU，使用PyTorch进行高速时间窗口切片。
# # # # # 5. (已修改) (鲁棒性) 增强了 LAS 加载逻辑，以处理缺少颜色的文件。

# # # # from plyfile import PlyData
# # # # import numpy as np
# # # # from tqdm import tqdm

# # # # import laspy
# # # # import torch
# # # # import datetime
# # # # from pathlib import Path
# # # # import pickle
# # # # import os
# # # # import sys
# # # # import traceback  # <-- (新增) 导入 traceback 以进行详细调试

# # # # # --- 项目路径设置 (与您的脚本保持一致) ---
# # # # current_dir = os.path.dirname(os.path.abspath(__file__))
# # # # parent_dir = os.path.dirname(current_dir)
# # # # grandparent_dir = os.path.dirname(parent_dir)
# # # # if grandparent_dir not in sys.path:
# # # #     sys.path.append(grandparent_dir)
# # # # from dji.opt import _get_opts


# # # # # --- 新增辅助函数，用于解析曝光时间字符串 ---
# # # # def parse_exposure_time(value):
# # # #     """(已修正) 将 ['1/2000'] 或 '1/2000' 这样的值转换为秒(float)"""
# # # #     # 检查输入是否为列表，如果是，则提取第一个元素
# # # #     if isinstance(value, list) and len(value) > 0:
# # # #         time_str = value[0]
# # # #     elif isinstance(value, str):
# # # #         time_str = value
# # # #     else:
# # # #         # 如果输入既不是列表也不是字符串，无法处理
# # # #         print(f"警告: 曝光时间格式未知 '{value}', 将使用默认值 0.001 秒。")
# # # #         return 0.001

# # # #     try:
# # # #         if '/' in time_str:
# # # #             num, den = map(float, time_str.split('/'))
# # # #             return num / den
# # # #         else:
# # # #             return float(time_str)
# # # #     except (ValueError, ZeroDivisionError, TypeError):
# # # #         # 增加了对TypeError的捕获，以防万一
# # # #         print(f"警告: 无法解析曝光时间字符串 '{time_str}', 将使用默认值 0.001 秒。")
# # # #         return 0.001


# # # # def main(hparams):
# # # #     # ==============================================================================
# # # #     # --- 新增配置：在这里选择您的点云划分方法 ---
# # # #     #
# # # #     # 'exposure_time' -> 使用每张照片的曝光时间 (最严谨，用于生成GT)。
# # # #     # 'fixed_window'  -> 使用下方定义的固定时间窗口 (点云更密集，用于场景理解)。
# # # #     SLICING_METHOD = 'fixed_window'

# # # #     # 当 SLICING_METHOD 设置为 'fixed_window' 时，此参数生效
# # # #     # 3.0 代表以照片时间戳为中心，前后各取1.5秒，总共3秒的窗口。
# # # #     FIXED_WINDOW_SECONDS = 14.0
# # # #     # ==============================================================================

# # # #     dataset_path = Path(hparams.dataset_path)
# # # #     las_output_path = Path(hparams.las_output_path)
# # # #     las_output_path.mkdir(parents=True, exist_ok=True)

# # # #     # --- 1. (已修改) 加载多个LiDAR点云数据块 ---
# # # #     print("正在扫描并加载 LAS/LAZ 文件...")
# # # #     # (修改) 我们复用 --las_path 参数，但将其视为一个目录
# # # #     las_dir = Path(hparams.las_path)
# # # #     las_files = sorted(list(las_dir.glob('*.las'))) + sorted(list(las_dir.glob('*.laz')))
    
# # # #     if not las_files:
# # # #         print(f"❌ 严重错误: 在 '{las_dir}' 中未找到任何 .las 或 .laz 文件。")
# # # #         sys.exit(1)
    
# # # #     print(f"找到 {len(las_files)} 个 LAS/LAZ 文件。")

# # # #     all_lidars = []
# # # #     all_gps_times = []
    
# # # #     # (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
# # # #     # --- (核心修复) 替换整个加载循环 ---
# # # #     # (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
# # # #     for las_file_path in tqdm(las_files, desc="Loading LAS/LAZ files"):
# # # #         try:
# # # #             with laspy.open(las_file_path) as in_file:
                
# # # #                 # (修复 1) 必须调用 .read() 来获取 LasData 对象
# # # #                 las = in_file.read()
                
# # # #                 # (修复 2) 检查必需的属性
# # # #                 if not (hasattr(las, 'x') and hasattr(las, 'y') and hasattr(las, 'z')):
# # # #                     print(f"警告: 文件 {las_file_path} 缺少 X, Y, 或 Z 坐标。跳过此文件。")
# # # #                     continue # 跳到下一个文件
                
# # # #                 if not hasattr(las, 'gps_time'):
# # # #                     print(f"警告: 文件 {las_file_path} 缺少 gps_time。跳过此文件。")
# # # #                     continue # 跳到下一个文件

# # # #                 x, y, z = las.x, las.y, las.z
# # # #                 num_points = len(x)

# # # #                 # (修复 3) 检查可选的颜色属性
# # # #                 if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
# # # #                     r = (las.red / 65536.0 * 255).astype(np.uint8)
# # # #                     g = (las.green / 65536.0 * 255).astype(np.uint8)
# # # #                     b = (las.blue / 65536.0 * 255).astype(np.uint8)
# # # #                 else:
# # # #                     # 如果缺少颜色，创建默认的白色 (255)
# # # #                     # print(f"注意: 文件 {las_file_path} 缺少颜色信息。将使用默认值 (255, 255, 255)。")
# # # #                     r = np.full(num_points, 255, dtype=np.uint8)
# # # #                     g = np.full(num_points, 255, dtype=np.uint8)
# # # #                     b = np.full(num_points, 255, dtype=np.uint8)
                
# # # #                 # (优化) 使用 np.stack 替换 list(zip(...))，速度更快
# # # #                 lidar_data = np.stack([x, y, z, r, g, b], axis=1)
                
# # # #                 all_lidars.append(lidar_data)
# # # #                 all_gps_times.append(np.array(las.gps_time))
                
# # # #         except Exception as e:
# # # #             # (修复 4) 打印详细的错误信息
# # # #             print(f"警告: 加载文件 {las_file_path} 时发生严重错误: {e}")
# # # #             print(traceback.format_exc()) # 打印完整的错误堆栈
    
# # # #     # (!!!) (!!!) (!!!) 
# # # #     # --- 修复结束 ---
# # # #     # (!!!) (!!!) (!!!) 


# # # #     # (新增) 在合并前检查列表是否为空
# # # #     if not all_lidars:
# # # #         print(f"❌ 严重错误: 未能从 {len(las_files)} 个文件中成功加载任何有效的点云数据。")
# # # #         print("请检查上面的警告信息，确认 LAS 文件是否包含 'x', 'y', 'z' 和 'gps_time' 字段。")
# # # #         sys.exit(1) # 退出脚本

# # # #     print("正在合并所有点云块...")
# # # #     lidars = np.concatenate(all_lidars, axis=0)
# # # #     gps_times = np.concatenate(all_gps_times, axis=0)
    
# # # #     # (新增) 必须按时间排序，以确保切片正确
# # # #     print(f"正在按 GPS 时间排序 {len(lidars)} 个点...")
# # # #     sort_indices = np.argsort(gps_times)
# # # #     lidars = lidars[sort_indices]
# # # #     gps_times = gps_times[sort_indices]
    
# # # #     diff_gps_time = gps_times - gps_times[0]
# # # #     print("LAS文件加载、合并、排序并计算相对时间戳完成。")


# # # #     # --- (新增) GPU 加速设置 ---
# # # #     device = None
# # # #     if torch.cuda.is_available():
# # # #         try:
# # # #             device = torch.device("cuda:0")
# # # #             print(f"\n--- 检测到 CUDA！正在尝试将 {len(lidars)} 个点移动到 GPU... ---")
# # # #             # 将数据上传到 GPU
# # # #             diff_gps_time_gpu = torch.tensor(diff_gps_time, device=device)
# # # #             lidars_gpu = torch.tensor(lidars, device=device)
# # # #             print("--- 数据成功移动到 GPU。将使用 GPU 进行切片。 ---")
# # # #         except Exception as e:
# # # #             print(f"--- GPU 错误 (可能内存不足 OOM): {e} ---")
# # # #             print("--- 将回退到 CPU (速度较慢)。 ---")
# # # #             device = torch.device("cpu")
# # # #             diff_gps_time_gpu = torch.tensor(diff_gps_time) # 存在 CPU
# # # #             lidars_gpu = torch.tensor(lidars)         # 存在 CPU
# # # #     else:
# # # #         print("\n--- 未检测到 CUDA。将使用 CPU。 ---")
# # # #         device = torch.device("cpu")
# # # #         diff_gps_time_gpu = torch.tensor(diff_gps_time)
# # # #         lidars_gpu = torch.tensor(lidars)

# # # #     # 释放CPU内存 (如果已成功上传到GPU)
# # # #     if device.type == 'cuda':
# # # #         del lidars
# # # #         del gps_times
# # # #         del diff_gps_time
# # # #         import gc
# # # #         gc.collect()
# # # #         print("--- 已释放 CPU 上的点云内存。 ---")


# # # #     # --- 2. (已修改) 加载并按EXIF时间排序照片元数据 ---
# # # #     # (修改) 移除 train/val 划分
# # # #     image_metadata_dir = dataset_path / 'image_metadata'
# # # #     if not image_metadata_dir.exists():
# # # #         print(f"❌ 严重错误: 找不到元数据目录 '{image_metadata_dir}'")
# # # #         print("请先运行第0个脚本。")
# # # #         sys.exit(1)
        
# # # #     all_metadata_paths = sorted(list(image_metadata_dir.iterdir()))

# # # #     print("\n正在加载所有图像元数据并按EXIF拍摄时间排序...")
# # # #     all_metadata = []
# # # #     for path in tqdm(all_metadata_paths, desc="Loading metadata"):
# # # #         # weights_only=False 是为了加载包含EXIF的dict (非安全)
# # # #         # 切换到 torch.load(path, map_location='cpu') 更安全，如果元数据是张量
# # # #         # 但看起来元数据是复杂的dict，所以 weights_only=False 是必须的
# # # #         metadata = torch.load(path, weights_only=False) 
# # # #         all_metadata.append({'data': metadata, 'path': path})

# # # #     all_metadata.sort(key=lambda x: x['data']['meta_tags']['EXIF DateTimeOriginal'].values)
# # # #     print("照片元数据排序完成。")

# # # #     # --- 3. 提取精确时间戳和曝光时间 ---
# # # #     print("\n正在提取每张照片的精确时间和曝光时间...")
# # # #     sorted_photo_info = []
# # # #     origin_timestamp = None

# # # #     for item in tqdm(all_metadata, desc="Extracting photo info"):
# # # #         metadata = item['data']
# # # #         time_str = metadata['meta_tags']['EXIF DateTimeOriginal'].values
# # # #         dt_local = datetime.datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
# # # #         unix_timestamp = dt_local.timestamp()

# # # #         if origin_timestamp is None:
# # # #             origin_timestamp = unix_timestamp
# # # #         time_difference = unix_timestamp - origin_timestamp

# # # #         exposure_time_key = 'EXIF ExposureTime'
# # # #         if exposure_time_key in metadata['meta_tags']:
# # # #             # 这里我们直接传递.values的原始输出（可能是列表）
# # # #             exposure_time_value = metadata['meta_tags'][exposure_time_key].values
# # # #             exposure_time_sec = parse_exposure_time(exposure_time_value)
# # # #         else:
# # # #             exposure_time_sec = 0.001

# # # #         sorted_photo_info.append({
# # # #             "relative_time": time_difference,
# # # #             "exposure_time": exposure_time_sec
# # # #         })

# # # #     # --- 4. (核心修改 & GPU加速) 根据选择的策略进行点云划分 ---
# # # #     print(f"\n当前划分策略: '{SLICING_METHOD}'")
# # # #     if SLICING_METHOD == 'fixed_window':
# # # #         print(f"固定窗口大小: {FIXED_WINDOW_SECONDS} 秒")

# # # #     points_lidar_list = []
# # # #     total_points_matched = 0
# # # #     non_empty_segments = 0

# # # #     for info in tqdm(sorted_photo_info, desc=f"Slicing point cloud (on {device.type})"):
# # # #         photo_time = info['relative_time']

# # # #         # 根据选择的策略，决定窗口的宽度
# # # #         if SLICING_METHOD == 'fixed_window':
# # # #             window_width = FIXED_WINDOW_SECONDS
# # # #         elif SLICING_METHOD == 'exposure_time':
# # # #             window_width = info['exposure_time']
# # # #         else:
# # # #             raise ValueError("SLICING_METHOD 必须是 'exposure_time' 或 'fixed_window'")

# # # #         # 计算时间窗口
# # # #         start_window = photo_time - (window_width / 2.0)
# # # #         end_window = photo_time + (window_width / 2.0)

# # # #         # --- (!!!) GPU 加速切片 (!!!) ---
# # # #         # 1. 在 GPU 上计算布尔掩码
# # # #         bool_mask = (diff_gps_time_gpu >= start_window) & (diff_gps_time_gpu < end_window)
        
# # # #         # 2. 在 GPU 上索引
# # # #         points_in_segment_gpu = lidars_gpu[bool_mask]

# # # #         # 3. (关键) 将结果移回 CPU 以便保存到列表和 pickle
# # # #         points_in_segment_cpu = points_in_segment_gpu.cpu().numpy()
# # # #         # --- (!!!) 加速结束 (!!!) ---

# # # #         # 更新统计
# # # #         num_points = len(points_in_segment_cpu)
# # # #         if num_points > 0:
# # # #             non_empty_segments += 1
# # # #             total_points_matched += num_points
# # # #         points_lidar_list.append(points_in_segment_cpu)

# # # #     # --- 5. 保存结果并生成总结报告 ---
# # # #     print("\n划分完成，正在保存结果...")
# # # #     output_pkl_path = las_output_path / 'points_lidar_list.pkl'
# # # #     with open(str(output_pkl_path), "wb") as file:
# # # #         pickle.dump(points_lidar_list, file)
# # # #     print(f"点云分段列表已保存至: {str(output_pkl_path)}")

# # # #     total_segments = len(points_lidar_list)
# # # #     print("\n" + "=" * 25 + " 最终匹配结果总结 " + "=" * 25)
# # # #     print(f"总共为 {total_segments} 张照片生成了时间分段。")
# # # #     print(f"其中有 {non_empty_segments} 个分段成功匹配到了点云 (非空)。")
# # # #     print(f"总共匹配到的点云数量为: {total_points_matched}")
# # # #     if total_segments > 0:
# # # #         avg_points = total_points_matched / non_empty_segments if non_empty_segments > 0 else 0
# # # #         print(f"平均每个非空分段包含 {avg_points:.1f} 个点。")

# # # #     if non_empty_segments > total_segments * 0.8 and total_points_matched > 1000:
# # # #         print("✅ 匹配情况良好！大部分时间段都找到了对应的点云。")
# # # #     elif non_empty_segments > 0:
# # # #         print("⚠️ 注意：部分时间段未能匹配到点云，请检查照片和LiDAR数据的时间覆盖范围是否完全重合。")
# # # #     else:
# # # #         print("❌ 严重错误：所有时间段都未能匹配到点云！请重新检查时间戳转换逻辑或数据源。")
# # # #     print("=" * 70)
# # # #     print('脚本执行完毕。')


# # # # if __name__ == '__main__':
# # # #     main(_get_opts())


# # # # 脚本功能：
# # # # 1. (已修改) 加载并按EXIF时间排序照片元数据 (移除了 train/val 划分)。
# # # # 2. (已修改) 支持加载一个目录下的多个 .las/.laz 文件块，并将其合并。
# # # # 3. (已修改) (核心修改) 提供两种点云划分策略 ('exposure_time', 'fixed_window')。
# # # # 4. (已修改) (GPU加速) 将点云上传GPU，并实现动态CPU回退逻辑以防止OOM。
# # # # 5. (已修改) (鲁棒性) 增强了 LAS 加载逻辑，以处理缺少颜色的文件。
# # # # 6. (已修复) 修复了 'weights_only' 在旧版 PyTorch 上的 TypeError。

# # # from plyfile import PlyData
# # # import numpy as np
# # # from tqdm import tqdm

# # # import laspy
# # # import torch
# # # import datetime
# # # from pathlib import Path
# # # import pickle
# # # import os
# # # import sys
# # # import traceback  # <-- (新增) 导入 traceback 以进行详细调试
# # # import gc # 导入垃圾回收

# # # # --- 项目路径设置 (与您的脚本保持一致) ---
# # # current_dir = os.path.dirname(os.path.abspath(__file__))
# # # parent_dir = os.path.dirname(current_dir)
# # # grandparent_dir = os.path.dirname(parent_dir)
# # # if grandparent_dir not in sys.path:
# # #     sys.path.append(grandparent_dir)
# # # from dji.opt import _get_opts


# # # # --- 新增辅助函数，用于解析曝光时间字符串 ---
# # # def parse_exposure_time(value):
# # #     """(已修正) 将 ['1/2000'] 或 '1/2000' 这样的值转换为秒(float)"""
# # #     # 检查输入是否为列表，如果是，则提取第一个元素
# # #     if isinstance(value, list) and len(value) > 0:
# # #         time_str = value[0]
# # #     elif isinstance(value, str):
# # #         time_str = value
# # #     else:
# # #         # 如果输入既不是列表也不是字符串，无法处理
# # #         print(f"警告: 曝光时间格式未知 '{value}', 将使用默认值 0.001 秒。")
# # #         return 0.001

# # #     try:
# # #         if '/' in time_str:
# # #             num, den = map(float, time_str.split('/'))
# # #             return num / den
# # #         else:
# # #             return float(time_str)
# # #     except (ValueError, ZeroDivisionError, TypeError):
# # #         # 增加了对TypeError的捕获，以防万一
# # #         print(f"警告: 无法解析曝光时间字符串 '{time_str}', 将使用默认值 0.001 秒。")
# # #         return 0.001

# # # # (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
# # # # --- (新增) OOM 保护配置 ---
# # # # (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
# # # # 如果单个切片（一张照片）匹配的点超过这个数量，
# # # # 将自动切换到 CPU 进行切片，以防止 GPU OOM。
# # # # 200,000,000 (2亿点) * 6 (dims) * 4 bytes/float32 = ~4.8 GB
# # # # 这是一个保守的估计，在 24GB 显卡上应该是安全的。
# # # MAX_POINTS_PER_SEGMENT_ON_GPU = 200_000_000


# # # def main(hparams):
# # #     # ==============================================================================
# # #     # --- 新增配置：在这里选择您的点云划分方法 ---
# # #     #
# # #     # 'exposure_time' -> 使用每张照片的曝光时间 (最严谨，用于生成GT)。
# # #     # 'fixed_window'  -> 使用下方定义的固定时间窗口 (点云更密集，用于场景理解)。
# # #     SLICING_METHOD = 'fixed_window'

# # #     # 当 SLICING_METHOD 设置为 'fixed_window' 时，此参数生效
# # #     # 3.0 代表以照片时间戳为中心，前后各取1.5秒，总共3秒的窗口。
# # #     FIXED_WINDOW_SECONDS = 140.0
# # #     # ==============================================================================

# # #     dataset_path = Path(hparams.dataset_path)
# # #     las_output_path = Path(hparams.las_output_path)
# # #     las_output_path.mkdir(parents=True, exist_ok=True)

# # #     # --- 1. (已修改) 加载多个LiDAR点云数据块 ---
# # #     print("正在扫描并加载 LAS/LAZ 文件...")
# # #     # (修改) 我们复用 --las_path 参数，但将其视为一个目录
# # #     las_dir = Path(hparams.las_path)
# # #     las_files = sorted(list(las_dir.glob('*.las'))) + sorted(list(las_dir.glob('*.laz')))
    
# # #     if not las_files:
# # #         print(f"❌ 严重错误: 在 '{las_dir}' 中未找到任何 .las 或 .laz 文件。")
# # #         sys.exit(1)
    
# # #     print(f"找到 {len(las_files)} 个 LAS/LAZ 文件。")

# # #     all_lidars = []
# # #     all_gps_times = []
    
# # #     # (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
# # #     # --- (核心修复) 替换整个加载循环 ---
# # #     # (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
# # #     for las_file_path in tqdm(las_files, desc="Loading LAS/LAZ files"):
# # #         try:
# # #             with laspy.open(las_file_path) as in_file:
                
# # #                 # (修复 1) 必须调用 .read() 来获取 LasData 对象
# # #                 las = in_file.read()
                
# # #                 # (修复 2) 检查必需的属性
# # #                 if not (hasattr(las, 'x') and hasattr(las, 'y') and hasattr(las, 'z')):
# # #                     print(f"警告: 文件 {las_file_path} 缺少 X, Y, 或 Z 坐标。跳过此文件。")
# # #                     continue # 跳到下一个文件
                
# # #                 if not hasattr(las, 'gps_time'):
# # #                     print(f"警告: 文件 {las_file_path} 缺少 gps_time。跳过此文件。")
# # #                     continue # 跳到下一个文件

# # #                 x, y, z = las.x, las.y, las.z
# # #                 num_points = len(x)

# # #                 # (修复 3) 检查可选的颜色属性
# # #                 if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
# # #                     # 假定颜色是 16-bit (0-65535)，转换为 8-bit (0-255)
# # #                     r = (las.red / 65536.0 * 255).astype(np.uint8)
# # #                     g = (las.green / 65536.0 * 255).astype(np.uint8)
# # #                     b = (las.blue / 65536.0 * 255).astype(np.uint8)
# # #                 else:
# # #                     # 如果缺少颜色，创建默认的白色 (255)
# # #                     # print(f"注意: 文件 {las_file_path} 缺少颜色信息。将使用默认值 (255, 255, 255)。")
# # #                     r = np.full(num_points, 255, dtype=np.uint8)
# # #                     g = np.full(num_points, 255, dtype=np.uint8)
# # #                     b = np.full(num_points, 255, dtype=np.uint8)
                
# # #                 # (优化) 使用 np.stack 替换 list(zip(...))，速度更快
# # #                 # 注意：x,y,z (float64), r,g,b (uint8)。
# # #                 # np.stack 会将 uint8 提升为 float64，这会占用更多内存
# # #                 # 我们先创建 float64 的 XYZ，再创建 uint8 的 RGB，然后合并
# # #                 # 为了简单和后续 torch.tensor 的一致性，我们统一转为 float32
                
# # #                 xyz = np.stack([x, y, z], axis=1).astype(np.float32)
# # #                 rgb = np.stack([r, g, b], axis=1).astype(np.float32) # 转为 float32 以便合并
# # #                 lidar_data = np.hstack((xyz, rgb))
                
# # #                 all_lidars.append(lidar_data)
# # #                 all_gps_times.append(np.array(las.gps_time).astype(np.float64)) # 时间戳用 float64
                
# # #         except Exception as e:
# # #             # (修复 4) 打印详细的错误信息
# # #             print(f"警告: 加载文件 {las_file_path} 时发生严重错误: {e}")
# # #             print(traceback.format_exc()) # 打印完整的错误堆栈
    
# # #     # (!!!) (!!!) (!!!) 
# # #     # --- 修复结束 ---
# # #     # (!!!) (!!!) (!!!) 


# # #     # (新增) 在合并前检查列表是否为空
# # #     if not all_lidars:
# # #         print(f"❌ 严重错误: 未能从 {len(las_files)} 个文件中成功加载任何有效的点云数据。")
# # #         print("请检查上面的警告信息，确认 LAS 文件是否包含 'x', 'y', 'z' 和 'gps_time' 字段。")
# # #         sys.exit(1) # 退出脚本

# # #     print("正在合并所有点云块...")
# # #     # (修改) 我们现在明确知道 all_lidars 是 float32, all_gps_times 是 float64
# # #     lidars_cpu = np.concatenate(all_lidars, axis=0) # (N, 6) float32
# # #     gps_times_cpu = np.concatenate(all_gps_times, axis=0) # (N,) float64
    
# # #     # 释放中间列表内存
# # #     del all_lidars
# # #     del all_gps_times
# # #     gc.collect()
    
# # #     # (新增) 必须按时间排序，以确保切片正确
# # #     print(f"正在按 GPS 时间排序 {len(lidars_cpu)} 个点...")
# # #     sort_indices = np.argsort(gps_times_cpu)
# # #     lidars_cpu = lidars_cpu[sort_indices]
# # #     gps_times_cpu = gps_times_cpu[sort_indices]
    
# # #     diff_gps_time_cpu = gps_times_cpu - gps_times_cpu[0]
# # #     print("LAS文件加载、合并、排序并计算相对时间戳完成。")


# # #     # --- (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
# # #     # --- (已修改) GPU 加速设置 (带 OOM 保护逻辑) ---
# # #     # --- (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
# # #     device = None
# # #     use_gpu = False
# # #     lidars_gpu = None
# # #     diff_gps_time_gpu = None

# # #     if torch.cuda.is_available():
# # #         try:
# # #             device = torch.device("cuda:0")
# # #             print(f"\n--- 检测到 CUDA！正在尝试将 {len(lidars_cpu)} 个点移动到 GPU... ---")
# # #             # 将数据上传到 GPU
# # #             # (N,) float64 -> (N,) float64 (GPU)
# # #             diff_gps_time_gpu = torch.tensor(diff_gps_time_cpu, device=device) 
# # #             # (N, 6) float32 -> (N, 6) float32 (GPU)
# # #             lidars_gpu = torch.tensor(lidars_cpu, device=device) 
# # #             print("--- 数据成功移动到 GPU。将使用 GPU 进行切片。 ---")
# # #             use_gpu = True
            
# # #             # (!!!) (修改) (!!!)
# # #             # 不再删除 CPU 副本，以便在 OOM 时回退
# # #             # del lidars_cpu
# # #             # del gps_times_cpu
# # #             # del diff_gps_time_cpu
# # #             # gc.collect()
# # #             print("--- 将在 CPU 和 GPU 中同时保留数据副本以用于 OOM 回退。 ---")
            
# # #         except Exception as e:
# # #             print(f"--- GPU 错误 (可能内存不足 OOM): {e} ---")
# # #             print("--- 将回退到 CPU (速度较慢)。 ---")
# # #             device = torch.device("cpu")
# # #             use_gpu = False
# # #             lidars_gpu = None
# # #             diff_gps_time_gpu = None
# # #     else:
# # #         print("\n--- 未检测到 CUDA。将使用 CPU。 ---")
# # #         device = torch.device("cpu")
# # #         use_gpu = False


# # #     # --- 2. (已修改) 加载并按EXIF时间排序照片元数据 ---
# # #     # (修改) 移除 train/val 划分
# # #     image_metadata_dir = dataset_path / 'image_metadata'
# # #     if not image_metadata_dir.exists():
# # #         print(f"❌ 严重错误: 找不到元数据目录 '{image_metadata_dir}'")
# # #         print("请先运行第0个脚本。")
# # #         sys.exit(1)
        
# # #     all_metadata_paths = sorted(list(image_metadata_dir.iterdir()))

# # #     print("\n正在加载所有图像元数据并按EXIF拍摄时间排序...")
# # #     all_metadata = []
# # #     for path in tqdm(all_metadata_paths, desc="Loading metadata"):
        
# # #         # (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
# # #         # --- (修复 1) 移除 weights_only=False ---
# # #         # (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
# # #         try:
# # #             # metadata = torch.load(path, weights_only=False) # <- 旧的、导致错误的代码
# # #             metadata = torch.load(path) # <- 修复后的代码
# # #             all_metadata.append({'data': metadata, 'path': path})
# # #         except Exception as e:
# # #             print(f"警告：加载元数据文件 {path} 失败: {e}")
# # #             print(traceback.format_exc())

# # #     all_metadata.sort(key=lambda x: x['data']['meta_tags']['EXIF DateTimeOriginal'].values)
# # #     print("照片元数据排序完成。")

# # #     # --- 3. 提取精确时间戳和曝光时间 ---
# # #     print("\n正在提取每张照片的精确时间和曝光时间...")
# # #     sorted_photo_info = []
# # #     origin_timestamp = None

# # #     for item in tqdm(all_metadata, desc="Extracting photo info"):
# # #         metadata = item['data']
# # #         time_str = metadata['meta_tags']['EXIF DateTimeOriginal'].values
# # #         dt_local = datetime.datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
# # #         unix_timestamp = dt_local.timestamp()

# # #         if origin_timestamp is None:
# # #             origin_timestamp = unix_timestamp
# # #         time_difference = unix_timestamp - origin_timestamp

# # #         exposure_time_key = 'EXIF ExposureTime'
# # #         if exposure_time_key in metadata['meta_tags']:
# # #             # 这里我们直接传递.values的原始输出（可能是列表）
# # #             exposure_time_value = metadata['meta_tags'][exposure_time_key].values
# # #             exposure_time_sec = parse_exposure_time(exposure_time_value)
# # #         else:
# # #             exposure_time_sec = 0.001

# # #         sorted_photo_info.append({
# # #             "relative_time": time_difference,
# # #             "exposure_time": exposure_time_sec
# # #         })

# # #     # --- 4. (核心修改 & GPU加速 & OOM 保护) 根据选择的策略进行点云划分 ---
# # #     print(f"\n当前划分策略: '{SLICING_METHOD}'")
# # #     if SLICING_METHOD == 'fixed_window':
# # #         print(f"固定窗口大小: {FIXED_WINDOW_SECONDS} 秒")
# # #     print(f"GPU OOM 保护阈值: {MAX_POINTS_PER_SEGMENT_ON_GPU} 个点/段")

# # #     points_lidar_list = []
# # #     total_points_matched = 0
# # #     non_empty_segments = 0
    
# # #     # (修改) 确保 tqdm 在 GPU 回退时能正确打印
# # #     pbar = tqdm(sorted_photo_info, desc=f"Slicing point cloud (on {device.type if use_gpu else 'cpu'})")

# # #     for info in pbar:
# # #         photo_time = info['relative_time']

# # #         # 根据选择的策略，决定窗口的宽度
# # #         if SLICING_METHOD == 'fixed_window':
# # #             window_width = FIXED_WINDOW_SECONDS
# # #         elif SLICING_METHOD == 'exposure_time':
# # #             window_width = info['exposure_time']
# # #         else:
# # #             raise ValueError("SLICING_METHOD 必须是 'exposure_time' 或 'fixed_window'")

# # #         # 计算时间窗口
# # #         start_window = photo_time - (window_width / 2.0)
# # #         end_window = photo_time + (window_width / 2.0)

# # #         # (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
# # #         # --- (修复 2) 动态 OOM 保护切片逻辑 ---
# # #         # (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
        
# # #         points_in_segment_cpu = None
# # #         num_points = 0

# # #         if use_gpu:
# # #             # 1. 在 GPU 上计算布尔掩码 (廉价)
# # #             bool_mask_gpu = (diff_gps_time_gpu >= start_window) & (diff_gps_time_gpu < end_window)
            
# # #             # 2. 在 GPU 上统计点数 (廉价)
# # #             num_points = torch.sum(bool_mask_gpu).item()
            
# # #             # 3. 检查是否超过 OOM 阈值
# # #             if num_points < MAX_POINTS_PER_SEGMENT_ON_GPU:
# # #                 # --- GPU 路径 (安全) ---
# # #                 points_in_segment_gpu = lidars_gpu[bool_mask_gpu]
# # #                 points_in_segment_cpu = points_in_segment_gpu.cpu().numpy()
# # #             else:
# # #                 # --- CPU 回退路径 (防止 OOM) ---
# # #                 pbar.write(f"⚠️ 警告: 段 {pbar.n} 匹配到 {num_points} 个点, "
# # #                            f"超过 {MAX_POINTS_PER_SEGMENT_ON_GPU} 阈值。"
# # #                            f" 正在回退到 CPU 进行切片...")
                
# # #                 # 使用 CPU 版本的 numpy 数组进行切片
# # #                 bool_mask_cpu = (diff_gps_time_cpu >= start_window) & (diff_gps_time_cpu < end_window)
# # #                 points_in_segment_cpu = lidars_cpu[bool_mask_cpu]
                
# # #         else:
# # #             # --- 纯 CPU 路径 ---
# # #             bool_mask_cpu = (diff_gps_time_cpu >= start_window) & (diff_gps_time_cpu < end_window)
# # #             points_in_segment_cpu = lidars_cpu[bool_mask_cpu]
# # #             num_points = len(points_in_segment_cpu)

# # #         # --- (!!!) 修复结束 (!!!) ---

# # #         # 更新统计
# # #         if num_points > 0:
# # #             non_empty_segments += 1
# # #             total_points_matched += num_points
# # #         points_lidar_list.append(points_in_segment_cpu)

# # #     # --- 5. 保存结果并生成总结报告 ---
# # #     print("\n划分完成，正在保存结果...")
# # #     output_pkl_path = las_output_path / 'points_lidar_list.pkl'
# # #     with open(str(output_pkl_path), "wb") as file:
# # #         pickle.dump(points_lidar_list, file)
# # #     print(f"点云分段列表已保存至: {str(output_pkl_path)}")

# # #     total_segments = len(points_lidar_list)
# # #     print("\n" + "=" * 25 + " 最终匹配结果总结 " + "=" * 25)
# # #     print(f"总共为 {total_segments} 张照片生成了时间分段。")
# # #     print(f"其中有 {non_empty_segments} 个分段成功匹配到了点云 (非空)。")
# # #     print(f"总共匹配到的点云数量为: {total_points_matched}")
# # #     if total_segments > 0:
# # #         avg_points = total_points_matched / non_empty_segments if non_empty_segments > 0 else 0
# # #         print(f"平均每个非空分段包含 {avg_points:.1f} 个点。")

# # #     if non_empty_segments > total_segments * 0.8 and total_points_matched > 1000:
# # #         print("✅ 匹配情况良好！大部分时间段都找到了对应的点云。")
# # #     elif non_empty_segments > 0:
# # #         print("⚠️ 注意：部分时间段未能匹配到点云，请检查照片和LiDAR数据的时间覆盖范围是否完全重合。")
# # #     else:
# # #         print("❌ 严重错误：所有时间段都未能匹配到点云！请重新检查时间戳转换逻辑或数据源。")
# # #     print("=" * 70)
# # #     print('脚本执行完毕。')


# # # if __name__ == '__main__':
# # #     main(_get_opts())


# # # 脚本功能：
# # # 1. (!!!) (核心修复) 增加了 LIDAR_TO_PHOTO_TIME_OFFSET 变量来调试时间戳不同步。
# # # 2. (!!!) (性能修复) 将 60GB 的 PKL 输出改为独立的 .npy 文件。
# # # 3. (保留) GPU 加速切片和 OOM 保护。

# # from plyfile import PlyData
# # import numpy as np
# # from tqdm import tqdm

# # import laspy
# # import torch
# # import datetime
# # from pathlib import Path
# # import pickle
# # import os
# # import sys
# # import traceback
# # import gc

# # # --- 项目路径设置 (与您的脚本保持一致) ---
# # current_dir = os.path.dirname(os.path.abspath(__file__))
# # parent_dir = os.path.dirname(current_dir)
# # grandparent_dir = os.path.dirname(parent_dir)
# # if grandparent_dir not in sys.path:
# #     sys.path.append(grandparent_dir)
# # from dji.opt import _get_opts


# # # --- 辅助函数 (保持不变) ---
# # def parse_exposure_time(value):
# #     # ... (代码与之前相同) ...
# #     if isinstance(value, list) and len(value) > 0:
# #         time_str = value[0]
# #     elif isinstance(value, str):
# #         time_str = value
# #     else:
# #         print(f"警告: 曝光时间格式未知 '{value}', 将使用默认值 0.001 秒。")
# #         return 0.001
# #     try:
# #         if '/' in time_str:
# #             num, den = map(float, time_str.split('/'))
# #             return num / den
# #         else:
# #             return float(time_str)
# #     except (ValueError, ZeroDivisionError, TypeError):
# #         print(f"警告: 无法解析曝光时间字符串 '{time_str}', 将使用默认值 0.001 秒。")
# #         return 0.001

# # # --- OOM 保护配置 (保持不变) ---
# # MAX_POINTS_PER_SEGMENT_ON_GPU = 200_000_000


# # def main(hparams):
# #     # ==============================================================================
# #     # --- (!!!) 核心调试：时间戳偏移 (!!!) ---
# #     #
# #     # 您的 'Projected=0' 问题 99% 在这里。
# #     # 调整此值以对齐 LiDAR 和照片时间。
# #     # 
# #     # (LiDAR 时间) + (OFFSET) = (照片时间)
# #     #
# #     # - 如果您认为 LiDAR 的时间戳比照片的 *早* 5 秒 (例如 LiDAR 10:00:00, 照片 10:00:05)
# #     #   那么 LiDAR 时间需要 *加上* 5 秒才能赶上照片。使用: OFFSET = 5.0
# #     #
# #     # - 如果您认为 LiDAR 的时间戳比照片的 *晚* 3 秒 (例如 LiDAR 10:00:08, 照片 10:00:05)
# #     #   那么 LiDAR 时间需要 *减去* 3 秒才能对齐照片。使用: OFFSET = -3.0
# #     #
# #     # (!!!) 调试工作流:
# #     # 1. 保持此值为 0.0 运行脚本 1。
# #     # 2. 运行脚本 2, 查看日志。如果 'Projected=0'。
# #     # 3. 猜测一个偏移量 (例如 5.0)，修改此值，重新运行脚本 1。
# #     # 4. 重新运行脚本 2, 查看 'Projected' 数量是否增加了。
# #     #
# #     LIDAR_TO_PHOTO_TIME_OFFSET = 0.0  # (!!!) 在此调试 (单位：秒) (!!!)
# #     # ==============================================================================
    
# #     # --- 配置：点云划分方法 ---
# #     SLICING_METHOD = 'fixed_window'
# #     FIXED_WINDOW_SECONDS = 600.0
# #     # ==============================================================================

# #     dataset_path = Path(hparams.dataset_path)
# #     las_output_path = Path(hparams.las_output_path)
# #     las_output_path.mkdir(parents=True, exist_ok=True)
    
# #     print(f"--- (!!!) 重要: 使用的时间戳偏移量 (LIDAR_TO_PHOTO_TIME_OFFSET) = {LIDAR_TO_PHOTO_TIME_OFFSET} 秒 ---")

# #     # --- 1. 加载 LiDAR 点云数据块 ---
# #     print("正在扫描并加载 LAS/LAZ 文件...")
# #     las_dir = Path(hparams.las_path) # (使用您修复后的 --las_path)
# #     # ... (加载逻辑与之前相同) ...
# #     las_files = sorted(list(las_dir.glob('*.las'))) + sorted(list(las_dir.glob('*.laz')))
# #     if not las_files:
# #         print(f"❌ 严重错误: 在 '{las_dir}' 中未找到任何 .las 或 .laz 文件。")
# #         sys.exit(1)
# #     print(f"找到 {len(las_files)} 个 LAS/LAZ 文件。")

# #     all_lidars = []
# #     all_gps_times = []
    
# #     for las_file_path in tqdm(las_files, desc="Loading LAS/LAZ files"):
# #         try:
# #             with laspy.open(las_file_path) as in_file:
# #                 las = in_file.read()
# #                 if not (hasattr(las, 'x') and hasattr(las, 'y') and hasattr(las, 'z') and hasattr(las, 'gps_time')):
# #                     print(f"警告: 文件 {las_file_path} 缺少 X, Y, Z, 或 gps_time。跳过。")
# #                     continue
# #                 x, y, z = las.x, las.y, las.z
# #                 num_points = len(x)
# #                 if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
# #                     r = (las.red / 65536.0 * 255).astype(np.uint8)
# #                     g = (las.green / 65536.0 * 255).astype(np.uint8)
# #                     b = (las.blue / 65536.0 * 255).astype(np.uint8)
# #                 else:
# #                     r = np.full(num_points, 255, dtype=np.uint8)
# #                     g = np.full(num_points, 255, dtype=np.uint8)
# #                     b = np.full(num_points, 255, dtype=np.uint8)
                
# #                 xyz = np.stack([x, y, z], axis=1).astype(np.float32)
# #                 rgb = np.stack([r, g, b], axis=1).astype(np.float32)
# #                 lidar_data = np.hstack((xyz, rgb))
# #                 all_lidars.append(lidar_data)
# #                 all_gps_times.append(np.array(las.gps_time).astype(np.float64))
# #         except Exception as e:
# #             print(f"警告: 加载文件 {las_file_path} 时发生严重错误: {e}")
# #             print(traceback.format_exc())
    
# #     if not all_lidars:
# #         print(f"❌ 严重错误: 未能从 {len(las_files)} 个文件中成功加载任何有效的点云数据。")
# #         sys.exit(1)

# #     print("正在合并所有点云块...")
# #     lidars_cpu = np.concatenate(all_lidars, axis=0) # (N, 6) float32
# #     gps_times_cpu = np.concatenate(all_gps_times, axis=0) # (N,) float64
# #     del all_lidars
# #     del all_gps_times
# #     gc.collect()
    
# #     print(f"正在按 GPS 时间排序 {len(lidars_cpu)} 个点...")
# #     sort_indices = np.argsort(gps_times_cpu)
# #     lidars_cpu = lidars_cpu[sort_indices]
# #     gps_times_cpu = gps_times_cpu[sort_indices]
    
# #     # (!!!) (核心修复) 我们只减去第一个 LiDAR 点的时间戳
# #     # 真正的对齐将在循环中通过 OFFSET 完成
# #     diff_gps_time_cpu = gps_times_cpu - gps_times_cpu[0]
# #     print("LAS文件加载、合并、排序并计算相对时间戳完成。")


# #     # --- GPU 加速设置 (带 OOM 保护逻辑) ---
# #     # ... (此部分与之前相同) ...
# #     device = None
# #     use_gpu = False
# #     lidars_gpu = None
# #     diff_gps_time_gpu = None
# #     if torch.cuda.is_available():
# #         try:
# #             device = torch.device("cuda:0")
# #             print(f"\n--- 检测到 CUDA！正在尝试将 {len(lidars_cpu)} 个点移动到 GPU... ---")
# #             diff_gps_time_gpu = torch.tensor(diff_gps_time_cpu, device=device) 
# #             lidars_gpu = torch.tensor(lidars_cpu, device=device) 
# #             print("--- 数据成功移动到 GPU。将使用 GPU 进行切片。 ---")
# #             use_gpu = True
# #             print("--- 将在 CPU 和 GPU 中同时保留数据副本以用于 OOM 回退。 ---")
# #         except Exception as e:
# #             print(f"--- GPU 错误 (可能内存不足 OOM): {e} ---")
# #             print("--- 将回退到 CPU (速度较慢)。 ---")
# #             device = torch.device("cpu")
# #             use_gpu = False
# #     else:
# #         print("\n--- 未检测到 CUDA。将使用 CPU。 ---")
# #         device = torch.device("cpu")
# #         use_gpu = False
# #     # ... (GPU 设置结束) ...

# #     # --- (!!!) (性能修复) 创建 .npy 输出目录 ---
# #     segment_output_dir = las_output_path / 'lidar_segments'
# #     segment_output_dir.mkdir(parents=True, exist_ok=True)
# #     print(f"点云分段将保存到: {segment_output_dir}")


# #     # --- 2. 加载并按EXIF时间排序照片元数据 ---
# #     image_metadata_dir = dataset_path / 'image_metadata'
# #     # ... (检查目录存在) ...
# #     all_metadata_paths = sorted(list(image_metadata_dir.iterdir()))
# #     print("\n正在加载所有图像元数据并按EXIF拍摄时间排序...")
# #     all_metadata = []
# #     for path in tqdm(all_metadata_paths, desc="Loading metadata"):
# #         try:
# #             metadata = torch.load(path) # (!!!) (已修复) 移除了 weights_only=False
# #             all_metadata.append({'data': metadata, 'path': path})
# #         except Exception as e:
# #             print(f"警告：加载元数据文件 {path} 失败: {e}")
# #             print(traceback.format_exc())

# #     all_metadata.sort(key=lambda x: x['data']['meta_tags']['EXIF DateTimeOriginal'].values)
# #     print("照片元数据排序完成。")

# #     # --- 3. 提取精确时间戳和曝光时间 ---
# #     print("\n正在提取每张照片的精确时间和曝光时间...")
# #     sorted_photo_info = []
# #     origin_timestamp = None
# #     for item in tqdm(all_metadata, desc="Extracting photo info"):
# #         metadata = item['data']
# #         time_str = metadata['meta_tags']['EXIF DateTimeOriginal'].values
# #         dt_local = datetime.datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
# #         unix_timestamp = dt_local.timestamp()

# #         if origin_timestamp is None:
# #             origin_timestamp = unix_timestamp
# #         # (!!!) (核心修复) 这是照片相对于照片[0]的时间
# #         time_difference = unix_timestamp - origin_timestamp

# #         exposure_time_key = 'EXIF ExposureTime'
# #         # ... (解析曝光时间) ...
# #         if exposure_time_key in metadata['meta_tags']:
# #             exposure_time_value = metadata['meta_tags'][exposure_time_key].values
# #             exposure_time_sec = parse_exposure_time(exposure_time_value)
# #         else:
# #             exposure_time_sec = 0.001

# #         sorted_photo_info.append({
# #             "relative_time": time_difference, # (这是照片的相对时间)
# #             "exposure_time": exposure_time_sec
# #         })

# #     # --- 4. (核心修改) 根据选择的策略进行点云划分 ---
# #     print(f"\n当前划分策略: '{SLICING_METHOD}'")
# #     if SLICING_METHOD == 'fixed_window':
# #         print(f"固定窗口大小: {FIXED_WINDOW_SECONDS} 秒")
# #     print(f"GPU OOM 保护阈值: {MAX_POINTS_PER_SEGMENT_ON_GPU} 个点/段")

# #     total_points_matched = 0
# #     non_empty_segments = 0
    
# #     pbar = tqdm(enumerate(sorted_photo_info), total=len(sorted_photo_info), desc=f"Slicing point cloud (on {device.type if use_gpu else 'cpu'})")

# #     for i, info in pbar:
# #         # (!!!) (核心修复) 这是照片相对于 照片[0] 的时间
# #         photo_time = info['relative_time']

# #         # (窗口宽度逻辑不变)
# #         if SLICING_METHOD == 'fixed_window':
# #             window_width = FIXED_WINDOW_SECONDS
# #         elif SLICING_METHOD == 'exposure_time':
# #             window_width = info['exposure_time']
# #         else:
# #             raise ValueError("SLICING_METHOD 必须是 'exposure_time' 或 'fixed_window'")

# #         # (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
# #         # --- (核心修复) 应用时间戳偏移量 ---
# #         # (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
        
# #         # 1. 计算照片的 [开始, 结束] 时间窗口 (相对于 照片[0])
# #         start_window_photo = photo_time - (window_width / 2.0)
# #         end_window_photo = photo_time + (window_width / 2.0)

# #         # 2. 将照片窗口时间平移到 LiDAR 的时间域
# #         #    (LiDAR 时间) + OFFSET = (照片时间)  =>  (LiDAR 时间) = (照片时间) - OFFSET
# #         start_window_lidar = start_window_photo - LIDAR_TO_PHOTO_TIME_OFFSET
# #         end_window_lidar = end_window_photo - LIDAR_TO_PHOTO_TIME_OFFSET
        
# #         # (切片逻辑使用 *lidar* 窗口)
# #         # (OOM 保护逻辑保持不变, 但使用新的 start/end_window_lidar)
# #         # ...
# #         points_in_segment_cpu = None
# #         num_points = 0
        
# #         if use_gpu:
# #             # (!!!) (核心修复) 使用 'start_window_lidar'
# #             bool_mask_gpu = (diff_gps_time_gpu >= start_window_lidar) & (diff_gps_time_gpu < end_window_lidar)
# #             num_points = torch.sum(bool_mask_gpu).item()
# #             if num_points < MAX_POINTS_PER_SEGMENT_ON_GPU:
# #                 points_in_segment_gpu = lidars_gpu[bool_mask_gpu]
# #                 points_in_segment_cpu = points_in_segment_gpu.cpu().numpy()
# #             else:
# #                 pbar.write(f"⚠️ 警告: 段 {pbar.n} 匹配到 {num_points} 个点 (OOM 保护), 回退到 CPU...")
# #                 # (!!!) (核心修复) 使用 'start_window_lidar'
# #                 bool_mask_cpu = (diff_gps_time_cpu >= start_window_lidar) & (diff_gps_time_cpu < end_window_lidar)
# #                 points_in_segment_cpu = lidars_cpu[bool_mask_cpu]
# #         else:
# #             # (!!!) (核心修复) 使用 'start_window_lidar'
# #             bool_mask_cpu = (diff_gps_time_cpu >= start_window_lidar) & (diff_gps_time_cpu < end_window_lidar)
# #             points_in_segment_cpu = lidars_cpu[bool_mask_cpu]
# #             num_points = len(points_in_segment_cpu)
# #         # ...
# #         # (切片逻辑结束)

# #         # (!!!) (性能修复) 保存为单独的 .npy 文件
# #         segment_filename = segment_output_dir / f'{i:06d}.npy'
# #         np.save(segment_filename, points_in_segment_cpu)

# #         # (统计逻辑保持不变)
# #         if num_points > 0:
# #             non_empty_segments += 1
# #             total_points_matched += num_points

# #     # --- 5. 保存结果并生成总结报告 ---
# #     print("\n划分完成，正在保存结果...")
# #     # (!!!) (性能修复) 移除了 pickle.dump
# #     print(f"点云分段已作为单独的 .npy 文件保存于: {segment_output_dir}")

# #     # (总结报告逻辑保持不变)
# #     total_segments = len(sorted_photo_info)
# #     print("\n" + "=" * 25 + " 最终匹配结果总结 " + "=" * 25)
# #     print(f"总共为 {total_segments} 张照片生成了时间分段。")
# #     print(f"其中有 {non_empty_segments} 个分段成功匹配到了点云 (非空)。")
# #     # ... (其余打印语句保持不变) ...
# #     print("=" * 70)
# #     print('脚本执行完毕。')


# # if __name__ == '__main__':
# #     main(_get_opts())


# # 脚本功能：
# # 1. (!!!) (核心修复) 递归加载所有 .las/.laz 文件。
# # 2. (!!!) (核心修复) 按 EXIF 时间排序所有照片。
# # 3. (!!!) (核心修复) 使用“相对时间对齐”，
# #    将 (min_las_time) 和 (min_photo_time) 对齐为 t=0。
# # 4. (保留) .npy 分段输出, GPU 加速, OOM 保护。

# import numpy as np
# from tqdm import tqdm
# import laspy
# import torch
# import datetime
# from pathlib import Path
# import pickle
# import os
# import sys
# import traceback
# import gc

# # --- 项目路径设置 (与您的脚本保持一致) ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# grandparent_dir = os.path.dirname(parent_dir)
# if grandparent_dir not in sys.path:
#     sys.path.append(grandparent_dir)
# from dji.opt import _get_opts


# # --- 辅助函数 (保持不变) ---
# def parse_exposure_time(value):
#     if isinstance(value, list) and len(value) > 0:
#         time_str = value[0]
#     elif isinstance(value, str):
#         time_str = value
#     else:
#         return 0.001
#     try:
#         if '/' in time_str:
#             num, den = map(float, time_str.split('/'))
#             return num / den
#         else:
#             return float(time_str)
#     except (ValueError, ZeroDivisionError, TypeError):
#         return 0.001

# # --- OOM 保护配置 (保持不变) ---
# MAX_POINTS_PER_SEGMENT_ON_GPU = 200_000_000


# def main(hparams):
#     # ==============================================================================
#     # --- (!!!) (核心修改) (!!!) ---
#     # 我们不再使用绝对时间偏移量 (LIDAR_TO_PHOTO_TIME_OFFSET)，
#     # 因为 LAS 时间戳 (1982年) 是无效的。
#     # 我们将使用“相对时间对齐”。
#     # ==============================================================================
    
#     # --- 配置：点云划分方法 ---
#     SLICING_METHOD = 'fixed_window'
    
#     # (!!!) 窗口大小
#     # 您的 600 秒窗口太大了。
#     # 对于 GT，您需要一个与照片曝光相匹配的小窗口。
#     # 10.0 秒意味着照片前后各 5 秒。
#     FIXED_WINDOW_SECONDS = 60.0
#     # ==============================================================================

#     dataset_path = Path(hparams.dataset_path)
#     las_output_path = Path(hparams.las_output_path)
#     las_output_path.mkdir(parents=True, exist_ok=True)
    
#     # --- 1. (!!!) (核心修复) 递归加载所有 LiDAR 点云数据块 ---
#     print("--- (1/4) 正在扫描并加载所有 LAS/LAZ 文件 (递归)... ---")
#     las_dir = Path(hparams.las_path)
    
#     # (!!!) (核心修复) 使用 .rglob() 来递归搜索所有子目录 (!!!)
#     las_files = sorted(list(las_dir.rglob('*.las'))) + sorted(list(las_dir.rglob('*.laz')))
    
#     if not las_files:
#         print(f"❌ 严重错误: 在 '{las_dir}' 及其子目录中未找到任何 .las 或 .laz 文件。")
#         sys.exit(1)
        
#     print(f"共找到 {len(las_files)} 个 LAS/LAZ 文件。")

#     all_lidars = []
#     all_gps_times = []
    
#     for las_file_path in tqdm(las_files, desc="加载 LAS 文件"):
#         try:
#             with laspy.open(las_file_path) as in_file:
#                 las = in_file.read()
#                 if not (hasattr(las, 'x') and hasattr(las, 'y') and hasattr(las, 'z') and hasattr(las, 'gps_time')):
#                     print(f"  警告: 文件 {las_file_path.name} 缺少 X, Y, Z, 或 gps_time。跳过。")
#                     continue
#                 x, y, z = las.x, las.y, las.z
#                 num_points = len(x)
#                 if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
#                     r = (las.red / 65536.0 * 255).astype(np.uint8)
#                     g = (las.green / 65536.0 * 255).astype(np.uint8)
#                     b = (las.blue / 65536.0 * 255).astype(np.uint8)
#                 else:
#                     r = np.full(num_points, 255, dtype=np.uint8)
#                     g = np.full(num_points, 255, dtype=np.uint8)
#                     b = np.full(num_points, 255, dtype=np.uint8)
                
#                 xyz = np.stack([x, y, z], axis=1).astype(np.float32)
#                 rgb = np.stack([r, g, b], axis=1).astype(np.float32)
#                 lidar_data = np.hstack((xyz, rgb))
#                 all_lidars.append(lidar_data)
#                 all_gps_times.append(np.array(las.gps_time).astype(np.float64))
#         except Exception as e:
#             print(f"  警告: 加载 {las_file_path.name} 失败: {e}")
    
#     if not all_lidars:
#         print(f"❌ 严重错误: 未能加载任何有效的点云数据。")
#         sys.exit(1)

#     print("正在合并所有点云块...")
#     lidars_cpu = np.concatenate(all_lidars, axis=0) # (N, 6) float32
#     gps_times_cpu = np.concatenate(all_gps_times, axis=0) # (N,) float64
#     del all_lidars
#     del all_gps_times
#     gc.collect()
    
#     print(f"正在按 GPS 时间排序 {len(lidars_cpu)} 个点...")
#     sort_indices = np.argsort(gps_times_cpu)
#     lidars_cpu = lidars_cpu[sort_indices]
#     gps_times_cpu = gps_times_cpu[sort_indices]
    
#     # (!!!) (核心修复) (!!!)
#     # --- 相对时间对齐：LiDAR ---
#     # 计算 LiDAR 的相对时间
#     min_las_time = gps_times_cpu[0]
#     relative_gps_time_cpu = gps_times_cpu - min_las_time # (!!!)
    
#     print("LAS文件加载、合并、排序并计算相对时间戳完成。")
#     print(f"  LiDAR 最小时间戳 (Raw): {min_las_time}")
#     print(f"  LiDAR 相对时间范围: 0.0 -> {relative_gps_time_cpu[-1]:.2f} 秒")


#     # --- (2/4) GPU 加速设置 (带 OOM 保护逻辑) ---
#     device = None
#     use_gpu = False
#     lidars_gpu = None
#     relative_gps_time_gpu = None # (!!!) (重命名) (!!!)
#     if torch.cuda.is_available():
#         try:
#             device = torch.device("cuda:0")
#             print(f"\n--- (2/4) 检测到 CUDA！正在尝试将 {len(lidars_cpu)} 个点移动到 GPU... ---")
            
#             # (!!!) (核心修复) 上传“相对”时间 (!!!)
#             relative_gps_time_gpu = torch.tensor(relative_gps_time_cpu, device=device) 
#             lidars_gpu = torch.tensor(lidars_cpu, device=device) 
            
#             print("  数据成功移动到 GPU。")
#             use_gpu = True
#             print("  将在 CPU 和 GPU 中同时保留数据副本以用于 OOM 回退。")
#         except Exception as e:
#             print(f"  GPU 错误 (OOM): {e}。将回退到 CPU。")
#             device = torch.device("cpu")
#             use_gpu = False
#     else:
#         print("\n--- (2/4) 未检测到 CUDA。将使用 CPU。 ---")
#         device = torch.device("cpu")
#         use_gpu = False

#     # --- (性能修复) 创建 .npy 输出目录 ---
#     segment_output_dir = las_output_path / 'lidar_segments'
#     segment_output_dir.mkdir(parents=True, exist_ok=True)
#     print(f"  点云分段将保存到: {segment_output_dir}")


#     # --- (3/4) (!!!) (核心修复) 加载并按 EXIF 时间排序照片 ---
#     image_metadata_dir = dataset_path / 'image_metadata'
#     if not image_metadata_dir.exists():
#         print(f"❌ 严重错误: 找不到元数据目录 '{image_metadata_dir}'")
#         sys.exit(1)
        
#     all_metadata_paths = sorted(list(image_metadata_dir.glob('*.pt')))
#     print(f"\n--- (3/4) 正在加载 {len(all_metadata_paths)} 个图像元数据... ---")
#     all_metadata = []
#     for path in tqdm(all_metadata_paths, desc="加载元数据"):
#         try:
#             metadata = torch.load(path)
#             # (!!!) (核心修复) 立即提取时间戳用于排序 (!!!)
#             time_str = metadata['meta_tags']['EXIF DateTimeOriginal'].values
#             dt_local = datetime.datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
#             unix_timestamp = dt_local.timestamp()
            
#             all_metadata.append({
#                 'data': metadata, 
#                 'path': path, 
#                 'absolute_time': unix_timestamp # (!!!)
#             })
#         except Exception as e:
#             print(f"  警告：加载元数据文件 {path.name} 失败: {e}")

#     if not all_metadata:
#         print(f"❌ 严重错误: 未能从 '{image_metadata_dir}' 加载任何元数据。")
#         sys.exit(1)

#     # (!!!) (核心修复) 必须按绝对时间排序 (!!!)
#     print("  按 EXIF 拍摄时间排序照片...")
#     all_metadata.sort(key=lambda x: x['absolute_time'])
#     print("  照片元数据排序完成。")

#     # (!!!) (核心修复) (!!!)
#     # --- 相对时间对齐：照片 ---
#     # 计算照片的相对时间
#     min_photo_time = all_metadata[0]['absolute_time']
#     sorted_photo_info = []
    
#     print("  正在计算照片的相对时间...")
#     for item in all_metadata:
#         # (!!!) (核心修复) 计算相对时间 (!!!)
#         relative_photo_time = item['absolute_time'] - min_photo_time
        
#         # (解析曝光时间 - 保持不变)
#         exposure_time_key = 'EXIF ExposureTime'
#         if exposure_time_key in item['data']['meta_tags']:
#             exposure_time_value = item['data']['meta_tags'][exposure_time_key].values
#             exposure_time_sec = parse_exposure_time(exposure_time_value)
#         else:
#             exposure_time_sec = 0.001

#         sorted_photo_info.append({
#             "relative_time": relative_photo_time, # (!!!) (这是照片的相对时间) (!!!)
#             "exposure_time": exposure_time_sec
#         })
        
#     print(f"  照片最小时间戳 (Raw): {min_photo_time}")
#     print(f"  照片相对时间范围: 0.0 -> {sorted_photo_info[-1]['relative_time']:.2f} 秒")


#     # --- (4/4) (!!!) (核心修复) 使用“相对时间”进行点云划分 ---
#     print(f"\n--- (4/4) 开始按“相对时间”划分点云... ---")
#     print(f"  划分策略: '{SLICING_METHOD}'")
#     print(f"  固定窗口大小: {FIXED_WINDOW_SECONDS} 秒")

#     total_points_matched = 0
#     non_empty_segments = 0
    
#     pbar = tqdm(enumerate(sorted_photo_info), total=len(sorted_photo_info), desc=f"Slicing (on {device.type if use_gpu else 'cpu'})")

#     for i, info in pbar:
#         # (!!!) (核心修复) 这是照片相对于 照片[0] 的时间
#         photo_relative_time = info['relative_time']

#         # (窗口宽度逻辑不变)
#         if SLICING_METHOD == 'fixed_window':
#             window_width = FIXED_WINDOW_SECONDS
#         elif SLICING_METHOD == 'exposure_time':
#             window_width = info['exposure_time']
        
#         # (!!!) (核心修复) (!!!)
#         # --- 相对时间窗口 ---
#         # 1. 计算照片的 [开始, 结束] 相对时间窗口
#         start_window_relative = photo_relative_time - (window_width / 2.0)
#         end_window_relative = photo_relative_time + (window_width / 2.0)

#         # 2. (已删除) 不再需要 LIDAR_TO_PHOTO_TIME_OFFSET
        
#         # (OOM 保护切片逻辑)
#         points_in_segment_cpu = None
#         num_points = 0
        
#         if use_gpu:
#             # (!!!) (核心修复) 使用 'relative_gps_time_gpu' 和 'start_window_relative'
#             bool_mask_gpu = (relative_gps_time_gpu >= start_window_relative) & (relative_gps_time_gpu < end_window_relative)
#             num_points = torch.sum(bool_mask_gpu).item()
            
#             if num_points < MAX_POINTS_PER_SEGMENT_ON_GPU:
#                 points_in_segment_gpu = lidars_gpu[bool_mask_gpu]
#                 points_in_segment_cpu = points_in_segment_gpu.cpu().numpy()
#             else:
#                 pbar.write(f"  ⚠️ 警告: 段 {i} 匹配到 {num_points} 个点 (OOM 保护), 回退到 CPU...")
#                 # (!!!) (核心修复) 使用 'relative_gps_time_cpu' 和 'start_window_relative'
#                 bool_mask_cpu = (relative_gps_time_cpu >= start_window_relative) & (relative_gps_time_cpu < end_window_relative)
#                 points_in_segment_cpu = lidars_cpu[bool_mask_cpu]
#         else:
#             # (!!!) (核心修复) 使用 'relative_gps_time_cpu' 和 'start_window_relative'
#             bool_mask_cpu = (relative_gps_time_cpu >= start_window_relative) & (relative_gps_time_cpu < end_window_relative)
#             points_in_segment_cpu = lidars_cpu[bool_mask_cpu]
#             num_points = len(points_in_segment_cpu)

#         # (!!!) (性能修复) 保存为单独的 .npy 文件
#         segment_filename = segment_output_dir / f'{i:06d}.npy'
#         np.save(segment_filename, points_in_segment_cpu)

#         # (统计逻辑保持不变)
#         if num_points > 0:
#             non_empty_segments += 1
#             total_points_matched += num_points

#     # --- 5. 保存结果并生成总结报告 ---
#     print("\n划分完成。")
#     print(f"点云分段已作为单独的 .npy 文件保存于: {segment_output_dir}")

#     total_segments = len(sorted_photo_info)
#     print("\n" + "=" * 25 + " 最终匹配结果总结 " + "=" * 25)
#     print(f"  (LiDAR 相对时间跨度: {relative_gps_time_cpu[-1]:.2f} 秒)")
#     print(f"  (照片相对时间跨度: {sorted_photo_info[-1]['relative_time']:.2f} 秒)")
#     print("-" * 25)
#     print(f"总共为 {total_segments} 张照片生成了时间分段。")
#     print(f"其中有 {non_empty_segments} 个分段成功匹配到了点云 (非空)。")
#     if total_segments > 0:
#         avg_points = total_points_matched / non_empty_segments if non_empty_segments > 0 else 0
#         print(f"平均每个非空分段包含 {avg_points:.1f} 个点。")

#     if non_empty_segments == 0:
#         print("❌ 严重错误：所有时间段都未能匹配到点云！")
#         print("   这很可能是因为您的 LiDAR 数据和照片数据来自完全不同的采集任务。")
#     elif non_empty_segments < total_segments * 0.5:
#         print("⚠️ 注意：只有一小部分照片匹配到了点云。")
#         print("   这可能是正常的，因为您的照片集跨越了多个航班/日期，")
#         print(f"   但您的 LiDAR 数据只覆盖了 {relative_gps_time_cpu[-1]:.2f} 秒的范围。")
#     else:
#          print("✅ 匹配情况良好！大部分时间段都找到了对应的点云。")
        
#     print("=" * 70)
#     print('脚本执行完毕。')


# if __name__ == '__main__':
#     main(_get_opts())

# 脚本功能：
# 1. 递归加载所有 .las/.laz 文件。
# 2. 按 EXIF 时间排序所有照片。
# 3. 使用“相对时间对齐”，将 (min_las_time) 和 (min_photo_time) 对齐为 t=0。
# 4. (!!!) (核心修复) (!!!)
#    保存 .npy 文件时，使用其对应的元数据 .pt 文件名 (例如 000123.npy)，
#    而不是使用 EXIF 排序的索引 (000000.npy)。
#    这确保了脚本 2 可以正确匹配位姿和点云。
# 5. (保留) .npy 分段输出, GPU 加速, OOM 保护。

import numpy as np
from tqdm import tqdm
import laspy
import torch
import datetime
from pathlib import Path
import pickle
import os
import sys
import traceback
import gc

# --- 项目路径设置 (与您的脚本保持一致) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)
from dji.opt import _get_opts


# --- 辅助函数 (保持不变) ---
def parse_exposure_time(value):
    if isinstance(value, list) and len(value) > 0:
        time_str = value[0]
    elif isinstance(value, str):
        time_str = value
    else:
        return 0.001
    try:
        if '/' in time_str:
            num, den = map(float, time_str.split('/'))
            return num / den
        else:
            return float(time_str)
    except (ValueError, ZeroDivisionError, TypeError):
        return 0.001

# --- OOM 保护配置 (保持不变) ---
MAX_POINTS_PER_SEGMENT_ON_GPU = 200_000_000


def main(hparams):
    # ==============================================================================
    # --- 相对时间对齐 ---
    # ==============================================================================
    
    # --- 配置：点云划分方法 ---
    SLICING_METHOD = 'fixed_window'
    
    # (!!!) 窗口大小
    # 10.0 秒意味着照片前后各 5 秒。
    FIXED_WINDOW_SECONDS = 80.0
    # ==============================================================================

    dataset_path = Path(hparams.dataset_path)
    las_output_path = Path(hparams.las_output_path)
    las_output_path.mkdir(parents=True, exist_ok=True)
    
    # --- 1. (!!!) (核心修复) 递归加载所有 LiDAR 点云数据块 ---
    print("--- (1/4) 正在扫描并加载所有 LAS/LAZ 文件 (递归)... ---")
    las_dir = Path(hparams.las_path)
    
    las_files = sorted(list(las_dir.rglob('*.las'))) + sorted(list(las_dir.rglob('*.laz')))
    
    if not las_files:
        print(f"❌ 严重错误: 在 '{las_dir}' 及其子目录中未找到任何 .las 或 .laz 文件。")
        sys.exit(1)
        
    print(f"共找到 {len(las_files)} 个 LAS/LAZ 文件。")

    all_lidars = []
    all_gps_times = []
    
    for las_file_path in tqdm(las_files, desc="加载 LAS 文件"):
        try:
            with laspy.open(las_file_path) as in_file:
                las = in_file.read()
                if not (hasattr(las, 'x') and hasattr(las, 'y') and hasattr(las, 'z') and hasattr(las, 'gps_time')):
                    print(f"  警告: 文件 {las_file_path.name} 缺少 X, Y, Z, 或 gps_time。跳过。")
                    continue
                x, y, z = las.x, las.y, las.z
                num_points = len(x)
                if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                    r = (las.red / 65536.0 * 255).astype(np.uint8)
                    g = (las.green / 65536.0 * 255).astype(np.uint8)
                    b = (las.blue / 65536.0 * 255).astype(np.uint8)
                else:
                    r = np.full(num_points, 255, dtype=np.uint8)
                    g = np.full(num_points, 255, dtype=np.uint8)
                    b = np.full(num_points, 255, dtype=np.uint8)
                
                xyz = np.stack([x, y, z], axis=1).astype(np.float32)
                rgb = np.stack([r, g, b], axis=1).astype(np.float32)
                lidar_data = np.hstack((xyz, rgb))
                all_lidars.append(lidar_data)
                all_gps_times.append(np.array(las.gps_time).astype(np.float64))
        except Exception as e:
            print(f"  警告: 加载 {las_file_path.name} 失败: {e}")
    
    if not all_lidars:
        print(f"❌ 严重错误: 未能加载任何有效的点云数据。")
        sys.exit(1)

    print("正在合并所有点云块...")
    lidars_cpu = np.concatenate(all_lidars, axis=0) # (N, 6) float32
    gps_times_cpu = np.concatenate(all_gps_times, axis=0) # (N,) float64
    del all_lidars
    del all_gps_times
    gc.collect()
    
    print(f"正在按 GPS 时间排序 {len(lidars_cpu)} 个点...")
    sort_indices = np.argsort(gps_times_cpu)
    lidars_cpu = lidars_cpu[sort_indices]
    gps_times_cpu = gps_times_cpu[sort_indices]
    
    # --- 相对时间对齐：LiDAR ---
    min_las_time = gps_times_cpu[0]
    relative_gps_time_cpu = gps_times_cpu - min_las_time # (!!!)
    
    print("LAS文件加载、合并、排序并计算相对时间戳完成。")
    print(f"  LiDAR 最小时间戳 (Raw): {min_las_time}")
    print(f"  LiDAR 相对时间范围: 0.0 -> {relative_gps_time_cpu[-1]:.2f} 秒")


    # --- (2/4) GPU 加速设置 (带 OOM 保护逻辑) ---
    device = None
    use_gpu = False
    lidars_gpu = None
    relative_gps_time_gpu = None 
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            print(f"\n--- (2/4) 检测到 CUDA！正在尝试将 {len(lidars_cpu)} 个点移动到 GPU... ---")
            
            relative_gps_time_gpu = torch.tensor(relative_gps_time_cpu, device=device) 
            lidars_gpu = torch.tensor(lidars_cpu, device=device) 
            
            print("  数据成功移动到 GPU。")
            use_gpu = True
            print("  将在 CPU 和 GPU 中同时保留数据副本以用于 OOM 回退。")
        except Exception as e:
            print(f"  GPU 错误 (OOM): {e}。将回退到 CPU。")
            device = torch.device("cpu")
            use_gpu = False
    else:
        print("\n--- (2/4) 未检测到 CUDA。将使用 CPU。 ---")
        device = torch.device("cpu")
        use_gpu = False

    # --- (性能修复) 创建 .npy 输出目录 ---
    segment_output_dir = las_output_path / 'lidar_segments'
    segment_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  点云分段将保存到: {segment_output_dir}")


    # --- (3/4) (!!!) (核心修复) 加载并按 EXIF 时间排序照片 ---
    image_metadata_dir = dataset_path / 'image_metadata'
    if not image_metadata_dir.exists():
        print(f"❌ 严重错误: 找不到元数据目录 '{image_metadata_dir}'")
        sys.exit(1)
        
    all_metadata_paths = sorted(list(image_metadata_dir.glob('*.pt')))
    print(f"\n--- (3/4) 正在加载 {len(all_metadata_paths)} 个图像元数据... ---")
    all_metadata = []
    for path in tqdm(all_metadata_paths, desc="加载元数据"):
        try:
            metadata = torch.load(path)
            time_str = metadata['meta_tags']['EXIF DateTimeOriginal'].values
            dt_local = datetime.datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
            unix_timestamp = dt_local.timestamp()
            
            all_metadata.append({
                'data': metadata, 
                'path': path, 
                'absolute_time': unix_timestamp 
            })
        except Exception as e:
            print(f"  警告：加载元数据文件 {path.name} 失败: {e}")

    if not all_metadata:
        print(f"❌ 严重错误: 未能从 '{image_metadata_dir}' 加载任何元数据。")
        sys.exit(1)

    print("  按 EXIF 拍摄时间排序照片...")
    all_metadata.sort(key=lambda x: x['absolute_time'])
    print("  照片元数据排序完成。")

    # --- 相对时间对齐：照片 ---
    min_photo_time = all_metadata[0]['absolute_time']
    sorted_photo_info = []
    
    print("  正在计算照片的相对时间...")
    # (!!!) (核心修复 1/3) (!!!)
    # 我们在 'all_metadata' (已按 EXIF 排序) 上循环
    for item in all_metadata: 
        relative_photo_time = item['absolute_time'] - min_photo_time
        
        # (!!!) (核心修复 1/3) (!!!)
        # 获取原始 .pt 文件的文件名词根 (例如 '000123')
        original_stem = item['path'].stem 
        
        exposure_time_key = 'EXIF ExposureTime'
        if exposure_time_key in item['data']['meta_tags']:
            exposure_time_value = item['data']['meta_tags'][exposure_time_key].values
            exposure_time_sec = parse_exposure_time(exposure_time_value)
        else:
            exposure_time_sec = 0.001

        sorted_photo_info.append({
            "relative_time": relative_photo_time, 
            "exposure_time": exposure_time_sec,
            "original_stem": original_stem  # (!!!) (核心修复 1/3) 存储词根
        })
        
    print(f"  照片最小时间戳 (Raw): {min_photo_time}")
    print(f"  照片相对时间范围: 0.0 -> {sorted_photo_info[-1]['relative_time']:.2f} 秒")


    # --- (4/4) (!!!) (核心修复) 使用“相对时间”进行点云划分 ---
    print(f"\n--- (4/4) 开始按“相对时间”划分点云... ---")
    print(f"  划分策略: '{SLICING_METHOD}'")
    print(f"  固定窗口大小: {FIXED_WINDOW_SECONDS} 秒")

    total_points_matched = 0
    non_empty_segments = 0
    
    # (!!!) (核心修复 2/3) (!!!)
    # 我们不再需要 enumerate，我们直接在 sorted_photo_info 上循环
    pbar = tqdm(sorted_photo_info, desc=f"Slicing (on {device.type if use_gpu else 'cpu'})")

    for info in pbar:
        photo_relative_time = info['relative_time']
        
        # (!!!) (核心修复 2/3) (!!!)
        # 获取此照片对应的原始文件名
        original_stem = info['original_stem']

        # (窗口宽度逻辑不变)
        if SLICING_METHOD == 'fixed_window':
            window_width = FIXED_WINDOW_SECONDS
        elif SLICING_METHOD == 'exposure_time':
            window_width = info['exposure_time']
        
        # --- 相对时间窗口 ---
        start_window_relative = photo_relative_time - (window_width / 2.0)
        end_window_relative = photo_relative_time + (window_width / 2.0)

        # (OOM 保护切片逻辑)
        points_in_segment_cpu = None
        num_points = 0
        
        if use_gpu:
            bool_mask_gpu = (relative_gps_time_gpu >= start_window_relative) & (relative_gps_time_gpu < end_window_relative)
            num_points = torch.sum(bool_mask_gpu).item()
            
            if num_points < MAX_POINTS_PER_SEGMENT_ON_GPU:
                points_in_segment_gpu = lidars_gpu[bool_mask_gpu]
                points_in_segment_cpu = points_in_segment_gpu.cpu().numpy()
            else:
                pbar.write(f"  ⚠️ 警告: 帧 {original_stem} 匹配到 {num_points} 个点 (OOM 保护), 回退到 CPU...")
                bool_mask_cpu = (relative_gps_time_cpu >= start_window_relative) & (relative_gps_time_cpu < end_window_relative)
                points_in_segment_cpu = lidars_cpu[bool_mask_cpu]
        else:
            bool_mask_cpu = (relative_gps_time_cpu >= start_window_relative) & (relative_gps_time_cpu < end_window_relative)
            points_in_segment_cpu = lidars_cpu[bool_mask_cpu]
            num_points = len(points_in_segment_cpu)

        # (!!!) (核心修复 3/3) (!!!)
        # 使用原始文件名词根来保存 .npy 文件
        segment_filename = segment_output_dir / f'{original_stem}.npy'
        np.save(segment_filename, points_in_segment_cpu)

        # (统计逻辑保持不变)
        if num_points > 0:
            non_empty_segments += 1
            total_points_matched += num_points

    # --- 5. 保存结果并生成总结报告 ---
    print("\n划分完成。")
    print(f"点云分段已作为单独的 .npy 文件保存于: {segment_output_dir}")
    
    # (总结报告逻辑不变)
    total_segments = len(sorted_photo_info)
    print("\n" + "=" * 25 + " 最终匹配结果总结 " + "=" * 25)
    print(f"  (LiDAR 相对时间跨度: {relative_gps_time_cpu[-1]:.2f} 秒)")
    print(f"  (照片相对时间跨度: {sorted_photo_info[-1]['relative_time']:.2f} 秒)")
    print("-" * 25)
    print(f"总共为 {total_segments} 张照片生成了时间分段。")
    print(f"其中有 {non_empty_segments} 个分段成功匹配到了点云 (非空)。")
    if total_segments > 0:
        avg_points = total_points_matched / non_empty_segments if non_empty_segments > 0 else 0
        print(f"平均每个非空分段包含 {avg_points:.1f} 个点。")

    if non_empty_segments == 0:
        print("❌ 严重错误：所有时间段都未能匹配到点云！")
        print("   这很可能是因为您的 LiDAR 数据和照片数据来自完全不同的采集任务。")
    elif non_empty_segments < total_segments * 0.5:
        print("⚠️ 注意：只有一小部分照片匹配到了点云。")
        print("   这可能是正常的，因为您的照片集跨越了多个航班/日期，")
        print(f"   但您的 LiDAR 数据只覆盖了 {relative_gps_time_cpu[-1]:.2f} 秒的范围。")
    else:
         print("✅ 匹配情况良好！大部分时间段都找到了对应的点云。")
        
    print("=" * 70)
    print('脚本执行完毕。')


if __name__ == '__main__':
    main(_get_opts())
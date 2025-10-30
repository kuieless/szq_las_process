# # # # 脚本功能：
# # # # 1. (已修改) 移除了 train/val 数据集划分。
# # # # 2. (已修改) (新增) 输出一个 projection_log.txt 文件，记录每张图像匹配和投影的点数。
# # # # 3. (已修改) (新增) 在关键步骤添加了详细的打印语句。
# # # # 4. 保留了核心的点云合并与投影逻辑。

# # # from plyfile import PlyData
# # # import numpy as np
# # # from tqdm import tqdm
# # # import xml.etree.ElementTree as ET
# # # import cv2
# # # import configargparse
# # # from pathlib import Path

# # # import json
# # # from PIL import Image
# # # from PIL.ExifTags import TAGS
# # # import exifread
# # # import math
# # # import torch
# # # import os
# # # import sys

# # # # --- 项目路径设置 (与您的脚本保持一致) ---
# # # current_dir = os.path.dirname(os.path.abspath(__file__))
# # # # 上一级目录: .../dji
# # # parent_dir = os.path.dirname(current_dir)
# # # # 上上一级目录: .../Aerial_lifting_early
# # # grandparent_dir = os.path.dirname(parent_dir)

# # # # 将 'Aerial_lifting_early' 目录添加到 sys.path
# # # if grandparent_dir not in sys.path:
# # #     sys.path.append(grandparent_dir)
# # # from dji.process_dji_v8_color import euler2rotation, rad
# # # from torchvision.utils import make_grid
# # # import os
# # # import pickle

# # # from dji.opt import _get_opts


# # # def main(hparams):
# # #     # ==============================================================================
# # #     # --- 1. 初始化、路径设置和日志文件 ---
# # #     # ==============================================================================
# # #     print("\n" + "=" * 20 + " 步骤 1: 初始化和路径设置 " + "=" * 20)
# # #     print(f"DEBUG 模式: {hparams.debug}")
# # #     down_scale = hparams.down_scale

# # #     dataset_path = Path(hparams.dataset_path)
# # #     original_images_path = Path(hparams.original_images_path)
# # #     las_output_path = Path(hparams.las_output_path)
# # #     output_path = Path(hparams.output_path) # Debug 输出路径

# # #     # (新增) 创建日志文件
# # #     log_file_path = dataset_path / 'projection_log.txt'
# # #     log_file = open(log_file_path, 'w')
    
# # #     print(f"数据集路径: {dataset_path}")
# # #     print(f"原始图像路径: {original_images_path}")
# # #     print(f"LiDAR pkl 路径: {las_output_path}")
# # #     print(f"日志文件将保存至: {log_file_path}")

# # #     log_file.write(f"--- 点云投影日志 ---\n")
# # #     log_file.write(f"数据集路径: {dataset_path}\n")
# # #     log_file.write(f"Debug 模式: {hparams.debug}\n")
# # #     log_file.write(f"降采样尺度: {down_scale}\n")
# # #     log_file.write(f"处理帧范围: {hparams.start} 到 {hparams.end}\n")
# # #     log_file.write("-" * 30 + "\n")

# # #     # ==============================================================================
# # #     # --- 2. 加载 XML, JSON, 和坐标数据 ---
# # #     # ==============================================================================
# # #     print("\n" + "=" * 20 + " 步骤 2: 加载元数据 " + "=" * 20)
# # #     # load the pose and image name from .xml file
# # #     root = ET.parse(hparams.infos_path).getroot()
# # #     xml_pose = np.array([[float(pose.find('Center/x').text),
# # #                           float(pose.find('Center/y').text),
# # #                           float(pose.find('Center/z').text),
# # #                           float(pose.find('Rotation/Omega').text),
# # #                           float(pose.find('Rotation/Phi').text),
# # #                           float(pose.find('Rotation/Kappa').text)] for pose in
# # #                          root.findall('Block/Photogroups/Photogroup/Photo/Pose')])
# # #     images_name = [Images_path.text.split("\\")[-1] for Images_path in
# # #                    root.findall('Block/Photogroups/Photogroup/Photo/ImagePath')]
# # #     print(f"从 XML 加载了 {len(images_name)} 条位姿记录。")

# # #     # load "original_images_list.json"
# # #     with open(hparams.original_images_list_json_path, "r") as file:
# # #         json_data = json.load(file)
# # #     print(f"从 JSON 加载了 {len(json_data)} 条图像记录。")

# # #     sorted_json_data = sorted(json_data, key=lambda x: x["origin_path"])
# # #     sorted_process_data = []

# # #     # (保持不变) 假设 XML 和 JSON 顺序一致
# # #     for i in tqdm(range(len(sorted_json_data)), desc="匹配 XML 与 JSON"):
# # #         sorted_process_line = {}
# # #         json_line = json_data[i]
# # #         sorted_process_line['images_name'] = images_name[i]
# # #         sorted_process_line['pose'] = xml_pose[i, :]
# # #         path_segments = json_line['origin_path'].split('/')
# # #         last_two_path = '/'.join(path_segments[-2:])
# # #         sorted_process_line['original_image_name'] = last_two_path
# # #         sorted_process_data.append(sorted_process_line)

# # #     xml_pose_sorted = np.array([x["pose"] for x in sorted_process_data])
# # #     images_name_sorted = [x["images_name"] for x in sorted_process_data]
# # #     original_image_name_sorted = [x["original_image_name"] for x in sorted_process_data]

# # #     # (保持不变) 加载相机参数
# # #     camera = np.array([float(root.findall('Block/Photogroups/Photogroup/FocalLengthPixels')[0].text),
# # #                        float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/x')[0].text),
# # #                        float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/y')[0].text)])
# # #     aspect_ratio = float(root.findall('Block/Photogroups/Photogroup/AspectRatio')[0].text)
# # #     camera_matrix_ori = np.array([[camera[0], 0, camera[1]],
# # #                                   [0, camera[0] * aspect_ratio, camera[2]],
# # #                                   [0, 0, 1]])

# # #     distortion_coeffs1 = np.array([float(hparams.k1),
# # #                                    float(hparams.k2),
# # #                                    float(hparams.p1),
# # #                                    float(hparams.p2),
# # #                                    float(hparams.k3)])  # k1 k2 p1 p2 k3
# # #     camera_matrix1 = np.array([[hparams.fx, 0, hparams.cx],
# # #                                [0, hparams.fx, hparams.cy],
# # #                                [0, 0, 1]])
# # #     print("相机内参和畸变参数已加载。")

# # #     # coordinate_info = torch.load(dataset_path / 'coordinates.pt', weights_only=False)
# # #     coordinate_info = torch.load(dataset_path / 'coordinates.pt')
# # #     origin_drb = coordinate_info['origin_drb'].numpy()
# # #     pose_scale_factor = coordinate_info['pose_scale_factor']
# # #     print("坐标系变换参数 (coordinates.pt) 已加载。")

# # #     # ==============================================================================
# # #     # --- 3. 加载点云并设置输出路径 ---
# # #     # ==============================================================================
# # #     print("\n" + "=" * 20 + " 步骤 3: 加载点云并设置路径 " + "=" * 20)
    
# # #     lidar_pkl_path = las_output_path / 'points_lidar_list.pkl'
# # #     with open(lidar_pkl_path, "rb") as file:
# # #         points_lidar_list = pickle.load(file)
# # #     print(f"成功从 {lidar_pkl_path} 加载 {len(points_lidar_list)} 个点云分段。")

# # #     # (!!!) (已修改) 移除 train/val 划分
# # #     metadata_dir = dataset_path / 'metadata'
# # #     depth_dir = dataset_path / 'depth_dji'
    
# # #     if hparams.debug:
# # #         (output_path / 'debug').mkdir(parents=True, exist_ok=True)
# # #         print(f"DEBUG 模式: 调试图像将保存到 {output_path / 'debug'}")
# # #     else:
# # #         depth_dir.mkdir(parents=True, exist_ok=True)
# # #         print(f"生产模式: 深度图 .npy 文件将保存到 {depth_dir}")
        
# # #     if not metadata_dir.exists():
# # #          print(f"❌ 严重错误: 找不到元数据目录 '{metadata_dir}'。请先运行脚本 0。")
# # #          sys.exit(1)

# # #     # ==============================================================================
# # #     # --- 4. 循环处理、投影并保存 ---
# # #     # ==============================================================================
# # #     print("\n" + "=" * 20 + " 步骤 4: 开始投影点云到图像 " + "=" * 20)

# # #     for i, rgb_name in enumerate(tqdm(images_name_sorted, desc="投影点云")):
# # #         if i < hparams.start or i >= hparams.end:
# # #             continue

# # #         # (已修改) 移除 debug 模式下的稀疏处理，以便完整调试
# # #         # if hparams.debug:
# # #         #     if i % 50 != 0 and not (i > (hparams.end - 4) or i < (hparams.start + 2)):
# # #         #         continue

# # #         # (!!!) (已修改) 移除 train/val 划分
# # #         # (原 split_dir 逻辑已删除)

# # #         img1 = cv2.imread(str(original_images_path) + '/' + original_image_name_sorted[i])
# # #         if img1 is None:
# # #             tqdm.write(f"警告: 无法读取图像 {original_image_name_sorted[i]}，跳过帧 {i}")
# # #             continue
            
# # #         img_change = cv2.undistort(img1, camera_matrix1, distortion_coeffs1, None, camera_matrix_ori)
# # #         img_change = cv2.resize(img_change, (
# # #             int(img_change.shape[1] / hparams.down_scale), int(img_change.shape[0] / hparams.down_scale)))
        
# # #         # (修改) 索引 points_lidar_list 应基于 i (全局索引)
# # #         # current_i = i - hparams.start # <-- 之前的逻辑
# # #         current_i = i # <-- 修正后的逻辑

# # #         if hparams.debug:
# # #             cv2.imwrite(str(output_path / 'debug' / f'{i:06d}_1_rgbs.png'), img_change)

# # #         # ==============================================================================
# # #         # --- 核心修正：使用更简洁和健壮的逻辑来合并前后帧的点云 ---
# # #         num_segments = len(points_lidar_list)

# # #         # 检查当前索引是否有效
# # #         if current_i >= num_segments:
# # #             tqdm.write(f"警告: 索引 {current_i} 超出点云列表范围 (大小: {num_segments})。跳过。")
# # #             continue
            
# # #         # 始终从当前帧的点云开始
# # #         points_nerf = points_lidar_list[current_i][:, :3].copy()
# # #         points_color = points_lidar_list[current_i][:, 3:].copy() if hparams.debug else None

# # #         # 如果前面还有帧，合并上一帧
# # #         if current_i - 1 >= 0:
# # #             points_nerf = np.concatenate((points_nerf, points_lidar_list[current_i - 1][:, :3]), axis=0)
# # #             if hparams.debug:
# # #                 points_color = np.concatenate((points_color, points_lidar_list[current_i - 1][:, 3:]), axis=0)

# # #         # 如果后面还有帧，合并下一帧
# # #         if current_i + 1 < num_segments:
# # #             points_nerf = np.concatenate((points_nerf, points_lidar_list[current_i + 1][:, :3]), axis=0)
# # #             if hparams.debug:
# # #                 points_color = np.concatenate((points_color, points_lidar_list[current_i + 1][:, 3:]), axis=0)
        
# # #         # (!!!) (新增) 记录总点数
# # #         total_points_matched = len(points_nerf)
# # #         # --- 修正结束 ---
# # #         # ==============================================================================

# # #         # Coordinate transformations
# # #         ZYQ = torch.DoubleTensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
# # #         ZYQ_1 = torch.DoubleTensor(
# # #             [[1, 0, 0], [0, math.cos(rad(135)), math.sin(rad(135))], [0, -math.sin(rad(135)), math.cos(rad(135))]])

# # #         points_nerf = ZYQ.numpy() @ points_nerf.T
# # #         points_nerf = (ZYQ_1.numpy() @ points_nerf).T
# # #         points_nerf = (points_nerf - origin_drb) / pose_scale_factor

# # #         # (!!!) (已修改) Load camera pose for the current frame
# # #         metadata_path = metadata_dir / f'{i:06d}.pt'
# # #         if not metadata_path.exists():
# # #             tqdm.write(f"警告: 元数据文件 {metadata_path} 未找到，跳过帧 {i}")
# # #             continue
# # #         metadata = torch.load(metadata_path, map_location='cpu')
        
# # #         camera_rotation = metadata['c2w'][:3, :3]
# # #         camera_position = metadata['c2w'][:3, 3]
# # #         camera_matrix = np.array([[metadata['intrinsics'][0] / down_scale, 0, metadata['intrinsics'][2] / down_scale],
# # #                                   [0, metadata['intrinsics'][1] / down_scale, metadata['intrinsics'][3] / down_scale],
# # #                                   [0, 0, 1]])

# # #         # Calculate world-to-camera transformation
# # #         E2 = np.hstack((camera_rotation, camera_position[:, np.newaxis]))
# # #         E2 = np.stack([E2[:, 0], E2[:, 1] * -1, E2[:, 2] * -1, E2[:, 3]], 1)
# # #         w2c = np.linalg.inv(np.concatenate((E2, [[0, 0, 0, 1]]), 0))

# # #         # Project points
# # #         points_homogeneous = np.hstack((points_nerf, np.ones((points_nerf.shape[0], 1))))
# # #         pt_3d_trans = np.dot(w2c, points_homogeneous.T)
# # #         pt_2d_trans = np.dot(camera_matrix, pt_3d_trans[:3])
        
# # #         # 避免除以零
# # #         pt_2d_trans[2, pt_2d_trans[2] == 0] = 1e-6
# # #         pt_2d_trans = pt_2d_trans / pt_2d_trans[2]
# # #         projected_points = (pt_2d_trans)[:2, :]

# # #         # Create depth map
# # #         large_int = 1e6
# # #         image_width, image_height = int(5472 / down_scale), int(3648 / down_scale)
# # #         depth_map = large_int * np.ones((image_height, image_width, 1), dtype=np.float32)

# # #         # Filter points that fall within the image plane
# # #         mask_x = np.logical_and(projected_points[0, :] >= 0, projected_points[0, :] < image_width)
# # #         mask_y = np.logical_and(projected_points[1, :] >= 0, projected_points[1, :] < image_height)
# # #         mask_z = pt_3d_trans[2, :] > 0  # Points in front of camera
# # #         mask = np.logical_and.reduce((mask_x, mask_y, mask_z))

# # #         # (!!!) (新增) 记录有效点数
# # #         effective_points_projected = np.sum(mask)

# # #         # (!!!) (新增) 写入日志
# # #         log_line = f"Image {i:06d}: Matched={total_points_matched:<10} | Projected={effective_points_projected:<10}"
# # #         log_file.write(log_line + "\n")
# # #         # 实时打印第一帧和 debug 模式下的日志
# # #         if hparams.debug or i == hparams.start:
# # #             tqdm.write(log_line)


# # #         projected_points = projected_points[:, mask]
# # #         depth_z = pt_3d_trans[2, mask]

# # #         # Populate depth map
# # #         for x, y, depth in zip(projected_points[0], projected_points[1], depth_z):
# # #             ix, iy = int(x), int(y)
# # #             if depth < depth_map[iy, ix]:
# # #                 depth_map[iy, ix] = depth

# # #         # (!!!) (已修改) Save or visualize results
# # #         if hparams.debug == False:
# # #             np.save(depth_dir / f'{i:06d}.npy', depth_map) # <-- 修改了保存路径
# # #         else:
# # #             # Visualization logic for debug mode
# # #             points_color_mask = points_color[mask]
# # #             image = 255 * np.ones((image_height, image_width, 3), dtype=np.uint8)
# # #             for x, y, depth, color in zip(projected_points[0], projected_points[1], depth_z, points_color_mask):
# # #                 ix, iy = int(x), int(y)
# # #                 if depth == depth_map[iy, ix]:  # Ensure only the closest point's color is used
# # #                     image[iy, ix] = color[::-1] # BGR -> RGB (或 RGB -> BGR，取决于输入)

# # #             # Create and save a colorized projection image
# # #             Image.fromarray(image).save(str(output_path / 'debug' / f'{i:06d}_2_project.png'))

# # #             # Create and save a depth map visualization
# # #             invalid_mask = (depth_map == large_int)
# # #             depth_map_valid = depth_map[~invalid_mask]
# # #             if depth_map_valid.size > 0:
# # #                 min_depth, max_depth = depth_map_valid.min(), depth_map_valid.max()
# # #                 depth_vis = depth_map.copy()
# # #                 depth_vis[invalid_mask] = min_depth # 使用 min_depth 填充无效区域以便归一化
# # #                 depth_vis = (depth_vis - min_depth) / max(max_depth - min_depth, 1e-8)
# # #                 depth_vis = ((1 - depth_vis) * 255).astype(np.uint8) # 反转深度，近白远黑
# # #                 depth_vis_color = cv2.applyColorMap(depth_vis.squeeze(), cv2.COLORMAP_JET)
# # #                 depth_vis_color[invalid_mask.squeeze()] = [255, 255, 255] # 无效区域设为白色
# # #                 Image.fromarray(depth_vis_color).save(str(output_path / 'debug' / f'{i:06d}_3_depth_filter.png'))

# # #     # ==============================================================================
# # #     # --- 5. 脚本结束 ---
# # #     # ==============================================================================
# # #     log_file.close()
# # #     print("\n" + "=" * 25 + " 脚本执行完毕 " + "=" * 25)
# # #     print(f"投影日志已保存到: {log_file_path}")


# # # if __name__ == '__main__':
# # #     main(_get_opts())

# # # 脚本功能：
# # # 1. (!!!) (性能修复) 按需加载 .npy 文件，而不是 60GB pkl。
# # # 2. (!!!) (性能修复) 使用 PyTorch 在 GPU 上执行点云投影。
# # # 3. (!!!) (性能修复) 移除缓慢的 Python 'for' 循环，使用 NumPy 'lexsort' 填充深度图。
# # # 4. (已修改) 移除了 train/val 划分。
# # # 5. (已修改) (新增) 输出 projection_log.txt。

# # from plyfile import PlyData
# # import numpy as np
# # from tqdm import tqdm
# # import xml.etree.ElementTree as ET
# # import cv2
# # import configargparse
# # from pathlib import Path

# # import json
# # from PIL import Image
# # from PIL.ExifTags import TAGS
# # import exifread
# # import math
# # import torch
# # import os
# # import sys

# # # --- 项目路径设置 (与您的脚本保持一致) ---
# # current_dir = os.path.dirname(os.path.abspath(__file__))
# # # ... (sys.path.append 逻辑保持不变) ...
# # parent_dir = os.path.dirname(current_dir)
# # grandparent_dir = os.path.dirname(parent_dir)
# # if grandparent_dir not in sys.path:
# #     sys.path.append(grandparent_dir)
    
# # from dji.process_dji_v8_color import euler2rotation, rad
# # from torchvision.utils import make_grid
# # import os
# # import pickle

# # from dji.opt import _get_opts


# # def main(hparams):
# #     # ==============================================================================
# #     # --- 1. 初始化、路径设置和日志文件 ---
# #     # ==============================================================================
# #     print("\n" + "=" * 20 + " 步骤 1: 初始化和路径设置 " + "=" * 20)
# #     print(f"DEBUG 模式: {hparams.debug}")
# #     down_scale = hparams.down_scale

# #     dataset_path = Path(hparams.dataset_path)
# #     original_images_path = Path(hparams.original_images_path)
# #     las_output_path = Path(hparams.las_output_path)
# #     output_path = Path(hparams.output_path) # Debug 输出路径

# #     # (新增) 创建日志文件
# #     log_file_path = dataset_path / 'projection_log.txt'
# #     log_file = open(log_file_path, 'w')
# #     log_file.write(f"--- 点云投影日志 (GPU 加速版) ---\n")
# #     log_file.write(f"数据集路径: {dataset_path}\n")
# #     log_file.write(f"Debug 模式: {hparams.debug}\n")
# #     log_file.write(f"降采样尺度: {down_scale}\n")
# #     log_file.write(f"处理帧范围: {hparams.start} 到 {hparams.end}\n")
# #     log_file.write("-" * 30 + "\n")

# #     # (!!!) (新增) (GPU 加速) 设置 GPU 设备
# #     device = None
# #     if torch.cuda.is_available():
# #         device = torch.device("cuda:0")
# #         print(f"--- 检测到 CUDA！将使用 GPU ({device}) 进行加速。 ---")
# #     else:
# #         device = torch.device("cpu")
# #         print("--- 未检测到 CUDA。将使用 CPU。 ---")

# #     # ==============================================================================
# #     # --- 2. 加载 XML, JSON, 和坐标数据 ---
# #     # ==============================================================================
# #     print("\n" + "=" * 20 + " 步骤 2: 加载元数据 " + "=" * 20)
# #     # ... (加载 XML, JSON, 匹配, 加载相机参数) ...
# #     root = ET.parse(hparams.infos_path).getroot()
# #     xml_pose = np.array([[float(pose.find('Center/x').text),
# #                           float(pose.find('Center/y').text),
# #                           float(pose.find('Center/z').text),
# #                           float(pose.find('Rotation/Omega').text),
# #                           float(pose.find('Rotation/Phi').text),
# #                           float(pose.find('Rotation/Kappa').text)] for pose in
# #                          root.findall('Block/Photogroups/Photogroup/Photo/Pose')])
# #     images_name = [Images_path.text.split("\\")[-1] for Images_path in
# #                    root.findall('Block/Photogroups/Photogroup/Photo/ImagePath')]
# #     print(f"从 XML 加载了 {len(images_name)} 条位姿记录。")

# #     with open(hparams.original_images_list_json_path, "r") as file:
# #         json_data = json.load(file)
# #     print(f"从 JSON 加载了 {len(json_data)} 条图像记录。")
# #     # ... (sorted_json_data, sorted_process_data 逻辑...)
# #     sorted_json_data = sorted(json_data, key=lambda x: x["origin_path"])
# #     sorted_process_data = []
# #     # (修复) 假设 XML 和 JSON 顺序一致
# #     # (如果它们不一致，您在 脚本 0 中的原始匹配逻辑是更好的，
# #     # 但为了与您的脚本 2 保持一致，我保留了索引匹配)
# #     for i in tqdm(range(len(sorted_json_data)), desc="匹配 XML 与 JSON"):
# #         sorted_process_line = {}
# #         json_line = json_data[i]
# #         sorted_process_line['images_name'] = images_name[i]
# #         sorted_process_line['pose'] = xml_pose[i, :]
# #         path_segments = json_line['origin_path'].split('/')
# #         last_two_path = '/'.join(path_segments[-2:])
# #         sorted_process_line['original_image_name'] = last_two_path
# #         sorted_process_data.append(sorted_process_line)

# #     xml_pose_sorted = np.array([x["pose"] for x in sorted_process_data])
# #     images_name_sorted = [x["images_name"] for x in sorted_process_data]
# #     original_image_name_sorted = [x["original_image_name"] for x in sorted_process_data]
    
# #     # ... (相机参数 camera, camera_matrix_ori, distortion_coeffs1, camera_matrix1...)
# #     camera = np.array([float(root.findall('Block/Photogroups/Photogroup/FocalLengthPixels')[0].text),
# #                        float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/x')[0].text),
# #                        float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/y')[0].text)])
# #     aspect_ratio = float(root.findall('Block/Photogroups/Photogroup/AspectRatio')[0].text)
# #     camera_matrix_ori = np.array([[camera[0], 0, camera[1]],
# #                                   [0, camera[0] * aspect_ratio, camera[2]],
# #                                   [0, 0, 1]])
# #     distortion_coeffs1 = np.array([float(hparams.k1),
# #                                    float(hparams.k2),
# #                                    float(hparams.p1),
# #                                    float(hparams.p2),
# #                                    float(hparams.k3)])
# #     camera_matrix1 = np.array([[hparams.fx, 0, hparams.cx],
# #                                [0, hparams.fx, hparams.cy],
# #                                [0, 0, 1]])
# #     print("相机内参和畸变参数已加载。")

# #     # (!!!) (修复) 修复 torch.load 错误
# #     coordinate_info = torch.load(dataset_path / 'coordinates.pt') # 移除了 weights_only
# #     origin_drb = coordinate_info['origin_drb'].numpy()
# #     pose_scale_factor = coordinate_info['pose_scale_factor']
# #     print("坐标系变换参数 (coordinates.pt) 已加载。")

# #     # ==============================================================================
# #     # --- 3. (!!!) (性能优化) 检查点云分段路径 ---
# #     # ==============================================================================
# #     print("\n" + "=" * 20 + " 步骤 3: 检查点云分段 " + "=" * 20)
    
# #     # (!!!) (性能优化) 我们不再加载 PKL 文件，而是指向 .npy 文件的目录
# #     segment_dir = las_output_path / 'lidar_segments'
    
# #     if not segment_dir.exists():
# #         print(f"❌ 严重错误: 找不到点云分段目录 '{segment_dir}'")
# #         print("请先运行修改后的脚本 1 (1_process_multi_las_gpu.py) 来生成 .npy 文件。")
# #         sys.exit(1)

# #     # (!!!) (性能优化) 统计文件数量以用于循环边界检查
# #     try:
# #         segment_files = sorted(list(segment_dir.glob('*.npy')))
# #         num_segments = len(segment_files)
# #         if num_segments == 0:
# #             raise FileNotFoundError
# #     except Exception as e:
# #         print(f"❌ 严重错误: 在 '{segment_dir}' 中找不到 .npy 分段文件。 {e}")
# #         sys.exit(1)
        
# #     print(f"在 {segment_dir} 中找到 {num_segments} 个点云分段 (.npy 文件)。")

# #     # (!!!) (性能优化) 移除了 60GB 的 pickle.load

# #     # (设置输出路径 depth_dir, metadata_dir 等的逻辑保持不变)
# #     metadata_dir = dataset_path / 'metadata'
# #     depth_dir = dataset_path / 'depth_dji'
    
# #     if hparams.debug:
# #         (output_path / 'debug').mkdir(parents=True, exist_ok=True)
# #         print(f"DEBUG 模式: 调试图像将保存到 {output_path / 'debug'}")
# #     else:
# #         depth_dir.mkdir(parents=True, exist_ok=True)
# #         print(f"生产模式: 深度图 .npy 文件将保存到 {depth_dir}")
        
# #     if not metadata_dir.exists():
# #          print(f"❌ 严重错误: 找不到元数据目录 '{metadata_dir}'。请先运行脚本 0。")
# #          sys.exit(1)

# #     # ==============================================================================
# #     # --- 4. 循环处理、投影并保存 ---
# #     # ==============================================================================
# #     print("\n" + "=" * 20 + " 步骤 4: 开始投影点云到图像 (GPU加速) " + "=" * 20)

# #     # (!!!) (新增) (GPU 加速) 将不变的矩阵移到 CPU/GPU
# #     ZYQ_cpu = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=torch.float64)
# #     r135 = rad(135)
# #     ZYQ_1_cpu = torch.tensor(
# #         [[1, 0, 0], [0, math.cos(r135), math.sin(r135)], [0, -math.sin(r135), math.cos(r135)]], dtype=torch.float64
# #     )
# #     origin_drb_gpu = torch.tensor(origin_drb, device=device, dtype=torch.float32)
    
# #     # (!!!) (新增) (GPU 加速)
# #     ZYQ = ZYQ_cpu.to(device=device, dtype=torch.float64)
# #     ZYQ_1 = ZYQ_1_cpu.to(device=device, dtype=torch.float64)


# #     for i, rgb_name in enumerate(tqdm(images_name_sorted, desc="投影点云")):
# #         if i < hparams.start or i >= hparams.end:
# #             continue

# #         # (图像加载和畸变校正逻辑保持不变)
# #         img1 = cv2.imread(str(original_images_path) + '/' + original_image_name_sorted[i])
# #         if img1 is None:
# #             tqdm.write(f"警告: 无法读取图像 {original_image_name_sorted[i]}，跳过帧 {i}")
# #             continue
# #         img_change = cv2.undistort(img1, camera_matrix1, distortion_coeffs1, None, camera_matrix_ori)
# #         img_change = cv2.resize(img_change, (
# #             int(img_change.shape[1] / hparams.down_scale), int(img_change.shape[0] / hparams.down_scale)))
        
# #         current_i = i # 当前帧索引

# #         if hparams.debug:
# #             cv2.imwrite(str(output_path / 'debug' / f'{i:06d}_1_rgbs.png'), img_change)

# #         # ==============================================================================
# #         # --- (!!!) (性能优化) 按需加载 .npy 文件 ---
# #         if current_i >= num_segments:
# #             tqdm.write(f"警告: 索引 {current_i} 超出点云分段数 ({num_segments})。跳过。")
# #             continue
# #         try:
# #             segment_path_current = segment_dir / f'{current_i:06d}.npy'
# #             points_data_current = np.load(segment_path_current)
# #         except FileNotFoundError:
# #             tqdm.write(f"警告: 找不到文件 {segment_path_current}。跳过帧 {i}")
# #             continue
# #         points_nerf_list = [points_data_current[:, :3].copy()]
# #         points_color_list = [points_data_current[:, 3:].copy()] if hparams.debug else None
# #         if current_i - 1 >= 0:
# #             try:
# #                 segment_path_prev = segment_dir / f'{current_i - 1:06d}.npy'
# #                 points_data_prev = np.load(segment_path_prev)
# #                 points_nerf_list.append(points_data_prev[:, :3])
# #                 if hparams.debug:
# #                     points_color_list.append(points_data_prev[:, 3:])
# #             except FileNotFoundError:
# #                 pass # 静默失败
# #         if current_i + 1 < num_segments:
# #             try:
# #                 segment_path_next = segment_dir / f'{current_i + 1:06d}.npy'
# #                 points_data_next = np.load(segment_path_next)
# #                 points_nerf_list.append(points_data_next[:, :3])
# #                 if hparams.debug:
# #                     points_color_list.append(points_data_next[:, 3:])
# #             except FileNotFoundError:
# #                 pass # 静默失败

# #         points_nerf_cpu = np.concatenate(points_nerf_list, axis=0)
# #         if hparams.debug:
# #             points_color = np.concatenate(points_color_list, axis=0)
# #         else:
# #             points_color = None
# #         # --- (性能优化) 结束 ---
# #         # ==============================================================================
        
# #         total_points_matched = len(points_nerf_cpu)
# #         if total_points_matched == 0:
# #             log_line = f"Image {i:06d}: Matched=0 | Projected=0"
# #             log_file.write(log_line + "\n")
# #             if hparams.debug or i == hparams.start:
# #                 tqdm.write(log_line)
# #             continue

# #         # (!!!) (新增) (GPU 加速) 坐标变换
# #         points_nerf_gpu = torch.tensor(points_nerf_cpu, device=device, dtype=torch.float64)
# #         points_nerf_gpu = ZYQ @ points_nerf_gpu.T
# #         points_nerf_gpu = (ZYQ_1 @ points_nerf_gpu).T
# #         points_nerf_gpu = points_nerf_gpu.to(dtype=torch.float32)
# #         points_nerf_gpu = (points_nerf_gpu - origin_drb_gpu) / pose_scale_factor

# #         # (加载相机位姿逻辑保持不变...)
# #         metadata_path = metadata_dir / f'{i:06d}.pt'
# #         if not metadata_path.exists():
# #             tqdm.write(f"警告: 元数据文件 {metadata_path} 未找到，跳过帧 {i}")
# #             continue
        
# #         # (!!!) (修复) 确保 metadata['c2w'] 是 float32
# #         metadata = torch.load(metadata_path, map_location='cpu')
# #         c2w_cpu = metadata['c2w'].to(dtype=torch.float32) 
# #         w2c_np = np.linalg.inv(c2w_cpu.numpy())
        
# #         camera_matrix_np = np.array([[metadata['intrinsics'][0] / down_scale, 0, metadata['intrinsics'][2] / down_scale],
# #                                      [0, metadata['intrinsics'][1] / down_scale, metadata['intrinsics'][3] / down_scale],
# #                                      [0, 0, 1]], dtype=np.float32)
        
# #         w2c_gpu = torch.tensor(w2c_np, device=device)
# #         camera_matrix_gpu = torch.tensor(camera_matrix_np, device=device)

# #         # (!!!) (新增) (GPU 加速) 投影点
# #         points_homogeneous_gpu = torch.cat(
# #             (points_nerf_gpu, torch.ones(total_points_matched, 1, device=device, dtype=torch.float32)), dim=1
# #         )
# #         pt_3d_trans_gpu = torch.matmul(w2c_gpu, points_homogeneous_gpu.T)
# #         pt_2d_trans_gpu = torch.matmul(camera_matrix_gpu, pt_3d_trans_gpu[:3])
        
# #         z = pt_2d_trans_gpu[2]
# #         z[z == 0] = 1e-6
# #         pt_2d_trans_gpu = pt_2d_trans_gpu / z
        
# #         projected_points_gpu = pt_2d_trans_gpu[:2]

# #         # (!!!) (新增) (GPU 加速) 过滤点
# #         image_width, image_height = int(5472 / down_scale), int(3648 / down_scale)
# #         mask_x_gpu = (projected_points_gpu[0, :] >= 0) & (projected_points_gpu[0, :] < image_width)
# #         mask_y_gpu = (projected_points_gpu[1, :] >= 0) & (projected_points_gpu[1, :] < image_height)
# #         mask_z_gpu = pt_3d_trans_gpu[2, :] > 0
# #         mask_gpu = mask_x_gpu & mask_y_gpu & mask_z_gpu
        
# #         effective_points_projected = torch.sum(mask_gpu).item()
# #         log_line = f"Image {i:06d}: Matched={total_points_matched:<10} | Projected={effective_points_projected:<10}"
# #         log_file.write(log_line + "\n")
# #         if hparams.debug or i == hparams.start:
# #             tqdm.write(log_line)
            
# #         if effective_points_projected == 0:
# #             if hparams.debug == False:
# #                  depth_map = 1e6 * np.ones((image_height, image_width, 1), dtype=np.float32)
# #                  np.save(depth_dir / f'{i:06d}.npy', depth_map)
# #             continue

# #         # (!!!) (新增) (GPU 加速) 将*过滤后*的点移回 CPU
# #         projected_points = projected_points_gpu[:, mask_gpu].cpu().numpy()
# #         depth_z = pt_3d_trans_gpu[2, mask_gpu].cpu().numpy()

# #         # (!!!) (新增) (矢量化) 使用 lexsort 填充深度图
# #         large_int = 1e6
# #         depth_map = large_int * np.ones((image_height, image_width, 1), dtype=np.float32)
# #         ix = projected_points[0].astype(np.int64)
# #         iy = projected_points[1].astype(np.int64)
# #         indices = np.lexsort((depth_z, ix, iy))
# #         sorted_ix = ix[indices]
# #         sorted_iy = iy[indices]
# #         sorted_depths = depth_z[indices]
# #         mask_unique = np.ones_like(sorted_ix, dtype=bool)
# #         mask_unique[1:] = (sorted_ix[1:] != sorted_ix[:-1]) | (sorted_iy[1:] != sorted_iy[:-1])
# #         final_ix = sorted_ix[mask_unique]
# #         final_iy = sorted_iy[mask_unique]
# #         final_depths = sorted_depths[mask_unique]
# #         depth_map[final_iy, final_ix] = final_depths[:, np.newaxis]
        
# #         # (!!!) (矢量化) 移除了缓慢的 'for' 循环

# #         # (保存或可视化结果 - 逻辑保持不变)
# #         if hparams.debug == False:
# #             np.save(depth_dir / f'{i:06d}.npy', depth_map)
# #         else:
# #             # (Debug 可视化逻辑)
# #             points_color_mask = points_color[mask_gpu.cpu().numpy()]
# #             image = 255 * np.ones((image_height, image_width, 3), dtype=np.uint8)
            
# #             # (!!!) (新增) (矢量化) 颜色填充
# #             sorted_colors = points_color_mask[indices]
# #             final_colors = sorted_colors[mask_unique]
# #             image[final_iy, final_ix] = final_colors[:, ::-1] # BGR -> RGB (或 RGB -> BGR)

# #             # (保存 debug 图像)
# #             Image.fromarray(image).save(str(output_path / 'debug' / f'{i:06d}_2_project.png'))
            
# #             # (深度图可视化逻辑保持不变)
# #             invalid_mask = (depth_map == large_int)
# #             depth_map_valid = depth_map[~invalid_mask]
# #             if depth_map_valid.size > 0:
# #                 min_depth, max_depth = depth_map_valid.min(), depth_map_valid.max()
# #                 depth_vis = depth_map.copy()
# #                 depth_vis[invalid_mask] = min_depth
# #                 depth_vis = (depth_vis - min_depth) / max(max_depth - min_depth, 1e-8)
# #                 depth_vis = ((1 - depth_vis) * 255).astype(np.uint8)
# #                 depth_vis_color = cv2.applyColorMap(depth_vis.squeeze(), cv2.COLORMAP_JET)
# #                 depth_vis_color[invalid_mask.squeeze()] = [255, 255, 255]
# #                 Image.fromarray(depth_vis_color).save(str(output_path / 'debug' / f'{i:06d}_3_depth_filter.png'))

# #     # (脚本结束逻辑保持不变)
# #     log_file.close()
# #     print("\n" + "=" * 25 + " 脚本执行完毕 " + "=" * 25)
# #     print(f"投影日志已保存到: {log_file_path}")


# # if __name__ == '__main__':
# #     main(_get_opts())

# # 脚本功能：
# # 1. (!!!) (性能修复) 按需加载 .npy 文件，而不是 60GB pkl。
# # 2. (!!!) (性能修复) 使用 PyTorch 在 GPU 上执行点云投影。
# # 3. (!!!) (性能修复) 移除缓慢的 Python 'for' 循环，使用 NumPy 'lexsort' 填充深度图。
# # 4. (已修改) 移除了 train/val 划分。
# # 5. (已修改) (新增) 输出 projection_log.txt。

# from plyfile import PlyData
# import numpy as np
# from tqdm import tqdm
# import xml.etree.ElementTree as ET
# import cv2
# import configargparse
# from pathlib import Path

# import json
# from PIL import Image
# from PIL.ExifTags import TAGS
# import exifread
# import math
# import torch
# import os
# import sys

# # --- 项目路径设置 (与您的脚本保持一致) ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # ... (sys.path.append 逻辑保持不变) ...
# parent_dir = os.path.dirname(current_dir)
# grandparent_dir = os.path.dirname(parent_dir)
# if grandparent_dir not in sys.path:
#     sys.path.append(grandparent_dir)
    
# from dji.process_dji_v8_color import euler2rotation, rad
# from torchvision.utils import make_grid
# import os
# import pickle

# from dji.opt import _get_opts


# def main(hparams):
#     # ==============================================================================
#     # --- 1. 初始化、路径设置和日志文件 ---
#     # ==============================================================================
#     print("\n" + "=" * 20 + " 步骤 1: 初始化和路径设置 " + "=" * 20)
#     print(f"DEBUG 模式: {hparams.debug}")
#     down_scale = hparams.down_scale

#     dataset_path = Path(hparams.dataset_path)
#     original_images_path = Path(hparams.original_images_path)
#     las_output_path = Path(hparams.las_output_path)
#     output_path = Path(hparams.output_path) # Debug 输出路径

#     # (新增) 创建日志文件
#     log_file_path = dataset_path / 'projection_log.txt'
#     log_file = open(log_file_path, 'w')
#     log_file.write(f"--- 点云投影日志 (GPU 加速版) ---\n")
#     log_file.write(f"数据集路径: {dataset_path}\n")
#     log_file.write(f"Debug 模式: {hparams.debug}\n")
#     log_file.write(f"降采样尺度: {down_scale}\n")
#     log_file.write(f"处理帧范围: {hparams.start} 到 {hparams.end}\n")
#     log_file.write("-" * 30 + "\n")

#     # (!!!) (新增) (GPU 加速) 设置 GPU 设备
#     device = None
#     if torch.cuda.is_available():
#         device = torch.device("cuda:0")
#         print(f"--- 检测到 CUDA！将使用 GPU ({device}) 进行加速。 ---")
#     else:
#         device = torch.device("cpu")
#         print("--- 未检测到 CUDA。将使用 CPU。 ---")

#     # ==============================================================================
#     # --- 2. 加载 XML, JSON, 和坐标数据 ---
#     # ==============================================================================
#     print("\n" + "=" * 20 + " 步骤 2: 加载元数据 " + "=" * 20)
#     # ... (加载 XML, JSON, 匹配, 加载相机参数) ...
#     root = ET.parse(hparams.infos_path).getroot()
#     xml_pose = np.array([[float(pose.find('Center/x').text),
#                           float(pose.find('Center/y').text),
#                           float(pose.find('Center/z').text),
#                           float(pose.find('Rotation/Omega').text),
#                           float(pose.find('Rotation/Phi').text),
#                           float(pose.find('Rotation/Kappa').text)] for pose in
#                          root.findall('Block/Photogroups/Photogroup/Photo/Pose')])
#     images_name = [Images_path.text.split("\\")[-1] for Images_path in
#                    root.findall('Block/Photogroups/Photogroup/Photo/ImagePath')]
#     print(f"从 XML 加载了 {len(images_name)} 条位姿记录。")

#     with open(hparams.original_images_list_json_path, "r") as file:
#         json_data = json.load(file)
#     print(f"从 JSON 加载了 {len(json_data)} 条图像记录。")
#     # ... (sorted_json_data, sorted_process_data 逻辑...)
#     sorted_json_data = sorted(json_data, key=lambda x: x["origin_path"])
#     sorted_process_data = []
#     # (修复) 假设 XML 和 JSON 顺序一致
#     # (如果它们不一致，您在 脚本 0 中的原始匹配逻辑是更好的，
#     # 但为了与您的脚本 2 保持一致，我保留了索引匹配)
#     for i in tqdm(range(len(sorted_json_data)), desc="匹配 XML 与 JSON"):
#         sorted_process_line = {}
#         json_line = json_data[i]
#         sorted_process_line['images_name'] = images_name[i]
#         sorted_process_line['pose'] = xml_pose[i, :]
#         path_segments = json_line['origin_path'].split('/')
#         last_two_path = '/'.join(path_segments[-2:])
#         sorted_process_line['original_image_name'] = last_two_path
#         sorted_process_data.append(sorted_process_line)

#     xml_pose_sorted = np.array([x["pose"] for x in sorted_process_data])
#     images_name_sorted = [x["images_name"] for x in sorted_process_data]
#     original_image_name_sorted = [x["original_image_name"] for x in sorted_process_data]
    
#     # ... (相机参数 camera, camera_matrix_ori, distortion_coeffs1, camera_matrix1...)
#     camera = np.array([float(root.findall('Block/Photogroups/Photogroup/FocalLengthPixels')[0].text),
#                        float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/x')[0].text),
#                        float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/y')[0].text)])
#     aspect_ratio = float(root.findall('Block/Photogroups/Photogroup/AspectRatio')[0].text)
#     camera_matrix_ori = np.array([[camera[0], 0, camera[1]],
#                                   [0, camera[0] * aspect_ratio, camera[2]],
#                                   [0, 0, 1]])
#     distortion_coeffs1 = np.array([float(hparams.k1),
#                                    float(hparams.k2),
#                                    float(hparams.p1),
#                                    float(hparams.p2),
#                                    float(hparams.k3)])
#     camera_matrix1 = np.array([[hparams.fx, 0, hparams.cx],
#                                [0, hparams.fx, hparams.cy],
#                                [0, 0, 1]])
#     print("相机内参和畸变参数已加载。")

#     # (!!!) (修复) 修复 torch.load 错误
#     coordinate_info = torch.load(dataset_path / 'coordinates.pt') # 移除了 weights_only
#     origin_drb = coordinate_info['origin_drb'].numpy()
#     pose_scale_factor = coordinate_info['pose_scale_factor']
#     print("坐标系变换参数 (coordinates.pt) 已加载。")

#     # ==============================================================================
#     # --- 3. (!!!) (性能优化) 检查点云分段路径 ---
#     # ==============================================================================
#     print("\n" + "=" * 20 + " 步骤 3: 检查点云分段 " + "=" * 20)
    
#     # (!!!) (性能优化) 我们不再加载 PKL 文件，而是指向 .npy 文件的目录
#     segment_dir = las_output_path / 'lidar_segments'
    
#     if not segment_dir.exists():
#         print(f"❌ 严重错误: 找不到点云分段目录 '{segment_dir}'")
#         print("请先运行修改后的脚本 1 (1_process_multi_las_gpu.py) 来生成 .npy 文件。")
#         sys.exit(1)

#     # (!!!) (性能优化) 统计文件数量以用于循环边界检查
#     try:
#         segment_files = sorted(list(segment_dir.glob('*.npy')))
#         num_segments = len(segment_files)
#         if num_segments == 0:
#             raise FileNotFoundError
#     except Exception as e:
#         print(f"❌ 严重错误: 在 '{segment_dir}' 中找不到 .npy 分段文件。 {e}")
#         sys.exit(1)
        
#     print(f"在 {segment_dir} 中找到 {num_segments} 个点云分段 (.npy 文件)。")

#     # (!!!) (性能优化) 移除了 60GB 的 pickle.load

#     # (设置输出路径 depth_dir, metadata_dir 等的逻辑保持不变)
#     metadata_dir = dataset_path / 'metadata'
#     depth_dir = dataset_path / 'depth_dji'
    
#     if hparams.debug:
#         (output_path / 'debug').mkdir(parents=True, exist_ok=True)
#         print(f"DEBUG 模式: 调试图像将保存到 {output_path / 'debug'}")
#     else:
#         depth_dir.mkdir(parents=True, exist_ok=True)
#         print(f"生产模式: 深度图 .npy 文件将保存到 {depth_dir}")
        
#     if not metadata_dir.exists():
#          print(f"❌ 严重错误: 找不到元数据目录 '{metadata_dir}'。请先运行脚本 0。")
#          sys.exit(1)

#     # ==============================================================================
#     # --- 4. 循环处理、投影并保存 ---
#     # ==============================================================================
#     print("\n" + "=" * 20 + " 步骤 4: 开始投影点云到图像 (GPU加速) " + "=" * 20)

#     # (!!!) (新增) (GPU 加速) 将不变的矩阵移到 CPU/GPU
#     ZYQ_cpu = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=torch.float64)
#     r135 = rad(135)
#     ZYQ_1_cpu = torch.tensor(
#         [[1, 0, 0], [0, math.cos(r135), math.sin(r135)], [0, -math.sin(r135), math.cos(r135)]], dtype=torch.float64
#     )
#     origin_drb_gpu = torch.tensor(origin_drb, device=device, dtype=torch.float32)
    
#     # (!!!) (新增) (GPU 加速)
#     ZYQ = ZYQ_cpu.to(device=device, dtype=torch.float64)
#     ZYQ_1 = ZYQ_1_cpu.to(device=device, dtype=torch.float64)


#     for i, rgb_name in enumerate(tqdm(images_name_sorted, desc="投影点云")):
#         if i < hparams.start or i >= hparams.end:
#             continue

#         # (图像加载和畸变校正逻辑保持不变)
#         img1 = cv2.imread(str(original_images_path) + '/' + original_image_name_sorted[i])
#         if img1 is None:
#             tqdm.write(f"警告: 无法读取图像 {original_image_name_sorted[i]}，跳过帧 {i}")
#             continue
#         img_change = cv2.undistort(img1, camera_matrix1, distortion_coeffs1, None, camera_matrix_ori)
#         img_change = cv2.resize(img_change, (
#             int(img_change.shape[1] / hparams.down_scale), int(img_change.shape[0] / hparams.down_scale)))
        
#         current_i = i # 当前帧索引

#         if hparams.debug:
#             cv2.imwrite(str(output_path / 'debug' / f'{i:06d}_1_rgbs.png'), img_change)

#         # ==============================================================================
#         # --- (!!!) (性能优化) 按需加载 .npy 文件 ---
#         if current_i >= num_segments:
#             tqdm.write(f"警告: 索引 {current_i} 超出点云分段数 ({num_segments})。跳过。")
#             continue
#         try:
#             segment_path_current = segment_dir / f'{current_i:06d}.npy'
#             points_data_current = np.load(segment_path_current)
#         except FileNotFoundError:
#             tqdm.write(f"警告: 找不到文件 {segment_path_current}。跳过帧 {i}")
#             continue
#         points_nerf_list = [points_data_current[:, :3].copy()]
#         points_color_list = [points_data_current[:, 3:].copy()] if hparams.debug else None
#         if current_i - 1 >= 0:
#             try:
#                 segment_path_prev = segment_dir / f'{current_i - 1:06d}.npy'
#                 points_data_prev = np.load(segment_path_prev)
#                 points_nerf_list.append(points_data_prev[:, :3])
#                 if hparams.debug:
#                     points_color_list.append(points_data_prev[:, 3:])
#             except FileNotFoundError:
#                 pass # 静默失败
#         if current_i + 1 < num_segments:
#             try:
#                 segment_path_next = segment_dir / f'{current_i + 1:06d}.npy'
#                 points_data_next = np.load(segment_path_next)
#                 points_nerf_list.append(points_data_next[:, :3])
#                 if hparams.debug:
#                     points_color_list.append(points_data_next[:, 3:])
#             except FileNotFoundError:
#                 pass # 静默失败

#         points_nerf_cpu = np.concatenate(points_nerf_list, axis=0)
#         if hparams.debug:
#             points_color = np.concatenate(points_color_list, axis=0)
#         else:
#             points_color = None
#         # --- (性能优化) 结束 ---
#         # ==============================================================================
        
#         total_points_matched = len(points_nerf_cpu)
#         if total_points_matched == 0:
#             log_line = f"Image {i:06d}: Matched=0 | Projected=0"
#             log_file.write(log_line + "\n")
#             if hparams.debug or i == hparams.start:
#                 tqdm.write(log_line)
#             continue

#         # (!!!) (新增) (GPU 加速) 坐标变换
#         points_nerf_gpu = torch.tensor(points_nerf_cpu, device=device, dtype=torch.float64)
#         points_nerf_gpu = ZYQ @ points_nerf_gpu.T
#         points_nerf_gpu = (ZYQ_1 @ points_nerf_gpu).T
#         points_nerf_gpu = points_nerf_gpu.to(dtype=torch.float32)
#         points_nerf_gpu = (points_nerf_gpu - origin_drb_gpu) / pose_scale_factor

#         # (加载相机位姿逻辑保持不变...)
#         metadata_path = metadata_dir / f'{i:06d}.pt'
#         if not metadata_path.exists():
#             tqdm.write(f"警告: 元数据文件 {metadata_path} 未找到，跳过帧 {i}")
#             continue
        
#         # (!!!) (修复) 确保 metadata['c2w'] 是 float32
#         metadata = torch.load(metadata_path, map_location='cpu')
#         c2w_cpu = metadata['c2w'].to(dtype=torch.float32) 
#         w2c_np = np.linalg.inv(c2w_cpu.numpy())
        
#         camera_matrix_np = np.array([[metadata['intrinsics'][0] / down_scale, 0, metadata['intrinsics'][2] / down_scale],
#                                      [0, metadata['intrinsics'][1] / down_scale, metadata['intrinsics'][3] / down_scale],
#                                      [0, 0, 1]], dtype=np.float32)
        
#         w2c_gpu = torch.tensor(w2c_np, device=device)
#         camera_matrix_gpu = torch.tensor(camera_matrix_np, device=device)

#         # (!!!) (新增) (GPU 加速) 投影点
#         points_homogeneous_gpu = torch.cat(
#             (points_nerf_gpu, torch.ones(total_points_matched, 1, device=device, dtype=torch.float32)), dim=1
#         )
#         pt_3d_trans_gpu = torch.matmul(w2c_gpu, points_homogeneous_gpu.T)
#         pt_2d_trans_gpu = torch.matmul(camera_matrix_gpu, pt_3d_trans_gpu[:3])
        
#         z = pt_2d_trans_gpu[2]
#         z[z == 0] = 1e-6
#         pt_2d_trans_gpu = pt_2d_trans_gpu / z
        
#         projected_points_gpu = pt_2d_trans_gpu[:2]

#         # (!!!) (新增) (GPU 加速) 过滤点
#         image_width, image_height = int(5472 / down_scale), int(3648 / down_scale)
#         mask_x_gpu = (projected_points_gpu[0, :] >= 0) & (projected_points_gpu[0, :] < image_width)
#         mask_y_gpu = (projected_points_gpu[1, :] >= 0) & (projected_points_gpu[1, :] < image_height)
#         mask_z_gpu = pt_3d_trans_gpu[2, :] > 0
#         mask_gpu = mask_x_gpu & mask_y_gpu & mask_z_gpu
        
#         effective_points_projected = torch.sum(mask_gpu).item()
#         log_line = f"Image {i:06d}: Matched={total_points_matched:<10} | Projected={effective_points_projected:<10}"
#         log_file.write(log_line + "\n")
#         if hparams.debug or i == hparams.start:
#             tqdm.write(log_line)
            
#         if effective_points_projected == 0:
#             if hparams.debug == False:
#                  depth_map = 1e6 * np.ones((image_height, image_width, 1), dtype=np.float32)
#                  np.save(depth_dir / f'{i:06d}.npy', depth_map)
#             continue

#         # (!!!) (新增) (GPU 加速) 将*过滤后*的点移回 CPU
#         projected_points = projected_points_gpu[:, mask_gpu].cpu().numpy()
#         depth_z = pt_3d_trans_gpu[2, mask_gpu].cpu().numpy()

#         # (!!!) (新增) (矢量化) 使用 lexsort 填充深度图
#         large_int = 1e6
#         depth_map = large_int * np.ones((image_height, image_width, 1), dtype=np.float32)
#         ix = projected_points[0].astype(np.int64)
#         iy = projected_points[1].astype(np.int64)
#         indices = np.lexsort((depth_z, ix, iy))
#         sorted_ix = ix[indices]
#         sorted_iy = iy[indices]
#         sorted_depths = depth_z[indices]
#         mask_unique = np.ones_like(sorted_ix, dtype=bool)
#         mask_unique[1:] = (sorted_ix[1:] != sorted_ix[:-1]) | (sorted_iy[1:] != sorted_iy[:-1])
#         final_ix = sorted_ix[mask_unique]
#         final_iy = sorted_iy[mask_unique]
#         final_depths = sorted_depths[mask_unique]
#         depth_map[final_iy, final_ix] = final_depths[:, np.newaxis]
        
#         # (!!!) (矢量化) 移除了缓慢的 'for' 循环

#         # (保存或可视化结果 - 逻辑保持不变)
#         if hparams.debug == False:
#             np.save(depth_dir / f'{i:06d}.npy', depth_map)
#         else:
#             # (Debug 可视化逻辑)
#             points_color_mask = points_color[mask_gpu.cpu().numpy()]
#             image = 255 * np.ones((image_height, image_width, 3), dtype=np.uint8)
            
#             # (!!!) (新增) (矢量化) 颜色填充
#             sorted_colors = points_color_mask[indices]
#             final_colors = sorted_colors[mask_unique]
#             image[final_iy, final_ix] = final_colors[:, ::-1] # BGR -> RGB (或 RGB -> BGR)

#             # (保存 debug 图像)
#             Image.fromarray(image).save(str(output_path / 'debug' / f'{i:06d}_2_project.png'))
            
#             # (深度图可视化逻辑保持不变)
#             invalid_mask = (depth_map == large_int)
#             depth_map_valid = depth_map[~invalid_mask]
#             if depth_map_valid.size > 0:
#                 min_depth, max_depth = depth_map_valid.min(), depth_map_valid.max()
#                 depth_vis = depth_map.copy()
#                 depth_vis[invalid_mask] = min_depth
#                 depth_vis = (depth_vis - min_depth) / max(max_depth - min_depth, 1e-8)
#                 depth_vis = ((1 - depth_vis) * 255).astype(np.uint8)
#                 depth_vis_color = cv2.applyColorMap(depth_vis.squeeze(), cv2.COLORMAP_JET)
#                 depth_vis_color[invalid_mask.squeeze()] = [255, 255, 255]
#                 Image.fromarray(depth_vis_color).save(str(output_path / 'debug' / f'{i:06d}_3_depth_filter.png'))

#     # (脚本结束逻辑保持不变)
#     log_file.close()
#     print("\n" + "=" * 25 + " 脚本执行完毕 " + "=" * 25)
#     print(f"投影日志已保存到: {log_file_path}")


# if __name__ == '__main__':
#     main(_get_opts())

# 脚本功能：
# 1. (!!!) (已修复) 修正了 'w2c' 矩阵求逆的 'LinAlgError' 错误。
# 2. (性能) 按需加载 .npy 文件，而不是 60GB pkl。
# 3. (性能) 使用 PyTorch 在 GPU 上执行点云投影。
# 4. (性能) 移除缓慢的 Python 'for' 循环，使用 NumPy 'lexsort' 填充深度图。
# 5. (已修改) 移除了 train/val 划分。
# 6. (已修改) (新增) 输出 projection_log.txt。

from plyfile import PlyData
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2
import configargparse
from pathlib import Path

import json
from PIL import Image
from PIL.ExifTags import TAGS
import exifread
import math
import torch
import os
import sys

# --- 项目路径设置 (与您的脚本保持一致) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# ... (sys.path.append 逻辑保持不变) ...
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)
    
from dji.process_dji_v8_color import euler2rotation, rad
from torchvision.utils import make_grid
import os
import pickle

from dji.opt import _get_opts


def main(hparams):
    # ==============================================================================
    # --- 1. 初始化、路径设置和日志文件 ---
    # ==============================================================================
    print("\n" + "=" * 20 + " 步骤 1: 初始化和路径设置 " + "=" * 20)
    print(f"DEBUG 模式: {hparams.debug}")
    down_scale = hparams.down_scale

    dataset_path = Path(hparams.dataset_path)
    original_images_path = Path(hparams.original_images_path)
    las_output_path = Path(hparams.las_output_path)
    output_path = Path(hparams.output_path) # Debug 输出路径

    # (新增) 创建日志文件
    log_file_path = dataset_path / 'projection_log.txt'
    log_file = open(log_file_path, 'w')
    log_file.write(f"--- 点云投影日志 (GPU 加速版) ---\n")
    log_file.write(f"数据集路径: {dataset_path}\n")
    log_file.write(f"Debug 模式: {hparams.debug}\n")
    log_file.write(f"降采样尺度: {down_scale}\n")
    log_file.write(f"处理帧范围: {hparams.start} 到 {hparams.end}\n")
    log_file.write("-" * 30 + "\n")

    # (!!!) (新增) (GPU 加速) 设置 GPU 设备
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"--- 检测到 CUDA！将使用 GPU ({device}) 进行加速。 ---")
    else:
        device = torch.device("cpu")
        print("--- 未检测到 CUDA。将使用 CPU。 ---")

    # ==============================================================================
    # --- 2. 加载 XML, JSON, 和坐标数据 ---
    # ==============================================================================
    print("\n" + "=" * 20 + " 步骤 2: 加载元数据 " + "=" * 20)
    # ... (加载 XML, JSON, 匹配, 加载相机参数) ...
    root = ET.parse(hparams.infos_path).getroot()
    xml_pose = np.array([[float(pose.find('Center/x').text),
                          float(pose.find('Center/y').text),
                          float(pose.find('Center/z').text),
                          float(pose.find('Rotation/Omega').text),
                          float(pose.find('Rotation/Phi').text),
                          float(pose.find('Rotation/Kappa').text)] for pose in
                         root.findall('Block/Photogroups/Photogroup/Photo/Pose')])
    images_name = [Images_path.text.split("\\")[-1] for Images_path in
                   root.findall('Block/Photogroups/Photogroup/Photo/ImagePath')]
    print(f"从 XML 加载了 {len(images_name)} 条位姿记录。")

    with open(hparams.original_images_list_json_path, "r") as file:
        json_data = json.load(file)
    print(f"从 JSON 加载了 {len(json_data)} 条图像记录。")
    # ... (sorted_json_data, sorted_process_data 逻辑...)
    sorted_json_data = sorted(json_data, key=lambda x: x["origin_path"])
    sorted_process_data = []
    # (修复) 假设 XML 和 JSON 顺序一致
    for i in tqdm(range(len(sorted_json_data)), desc="匹配 XML 与 JSON"):
        sorted_process_line = {}
        json_line = json_data[i]
        sorted_process_line['images_name'] = images_name[i]
        sorted_process_line['pose'] = xml_pose[i, :]
        path_segments = json_line['origin_path'].split('/')
        last_two_path = '/'.join(path_segments[-2:])
        sorted_process_line['original_image_name'] = last_two_path
        sorted_process_data.append(sorted_process_line)

    xml_pose_sorted = np.array([x["pose"] for x in sorted_process_data])
    images_name_sorted = [x["images_name"] for x in sorted_process_data]
    original_image_name_sorted = [x["original_image_name"] for x in sorted_process_data]
    
    # ... (相机参数 camera, camera_matrix_ori, distortion_coeffs1, camera_matrix1...)
    camera = np.array([float(root.findall('Block/Photogroups/Photogroup/FocalLengthPixels')[0].text),
                       float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/x')[0].text),
                       float(root.findall('Block/Photogroups/Photogroup/PrincipalPoint/y')[0].text)])
    aspect_ratio = float(root.findall('Block/Photogroups/Photogroup/AspectRatio')[0].text)
    camera_matrix_ori = np.array([[camera[0], 0, camera[1]],
                                  [0, camera[0] * aspect_ratio, camera[2]],
                                  [0, 0, 1]])
    distortion_coeffs1 = np.array([float(hparams.k1),
                                   float(hparams.k2),
                                   float(hparams.p1),
                                   float(hparams.p2),
                                   float(hparams.k3)])
    camera_matrix1 = np.array([[hparams.fx, 0, hparams.cx],
                               [0, hparams.fx, hparams.cy],
                               [0, 0, 1]])
    print("相机内参和畸变参数已加载。")

    # (!!!) (修复) 修复 torch.load 错误
    coordinate_info = torch.load(dataset_path / 'coordinates.pt') # 移除了 weights_only
    origin_drb = coordinate_info['origin_drb'].numpy()
    pose_scale_factor = coordinate_info['pose_scale_factor']
    print("坐标系变换参数 (coordinates.pt) 已加载。")

    # ==============================================================================
    # --- 3. (!!!) (性能优化) 检查点云分段路径 ---
    # ==============================================================================
    print("\n" + "=" * 20 + " 步骤 3: 检查点云分段 " + "=" * 20)
    
    # (!!!) (性能优化) 我们不再加载 PKL 文件，而是指向 .npy 文件的目录
    segment_dir = las_output_path / 'lidar_segments'
    
    if not segment_dir.exists():
        print(f"❌ 严重错误: 找不到点云分段目录 '{segment_dir}'")
        print("请先运行修改后的脚本 1 (1_process_multi_las_gpu.py) 来生成 .npy 文件。")
        sys.exit(1)

    # (!!!) (性能优化) 统计文件数量以用于循环边界检查
    try:
        segment_files = sorted(list(segment_dir.glob('*.npy')))
        num_segments = len(segment_files)
        # (!!!) (修复) 检查 XML/JSON 的数量是否与分段数匹配
        if num_segments != len(images_name_sorted):
            print(f"⚠️ 警告: 元数据中的图像数 ({len(images_name_sorted)}) 与 .npy 分段数 ({num_segments}) 不匹配。")
            print("这可能是因为脚本 1 提前终止。脚本将只处理 {min(num_segments, len(images_name_sorted))} 帧。")
            num_segments = min(num_segments, len(images_name_sorted))

        if num_segments == 0:
            raise FileNotFoundError("目录中 .npy 文件数为 0")
            
    except Exception as e:
        print(f"❌ 严重错误: 在 '{segment_dir}' 中找不到 .npy 分段文件。 {e}")
        sys.exit(1)
        
    print(f"在 {segment_dir} 中找到 {num_segments} 个点云分段 (.npy 文件)。")

    # (!!!) (性能优化) 移除了 60GB 的 pickle.load

    # (设置输出路径 depth_dir, metadata_dir 等的逻辑保持不变)
    metadata_dir = dataset_path / 'metadata'
    depth_dir = dataset_path / 'depth_dji'
    
    if hparams.debug:
        (output_path / 'debug').mkdir(parents=True, exist_ok=True)
        print(f"DEBUG 模式: 调试图像将保存到 {output_path / 'debug'}")
    else:
        depth_dir.mkdir(parents=True, exist_ok=True)
        print(f"生产模式: 深度图 .npy 文件将保存到 {depth_dir}")
        
    if not metadata_dir.exists():
         print(f"❌ 严重错误: 找不到元数据目录 '{metadata_dir}'。请先运行脚本 0。")
         sys.exit(1)

    # ==============================================================================
    # --- 4. 循环处理、投影并保存 ---
    # ==============================================================================
    print("\n" + "=" * 20 + " 步骤 4: 开始投影点云到图像 (GPU加速) " + "=" * 20)

    # (!!!) (新增) (GPU 加速) 将不变的矩阵移到 CPU/GPU
    ZYQ_cpu = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=torch.float64)
    r135 = rad(135)
    ZYQ_1_cpu = torch.tensor(
        [[1, 0, 0], [0, math.cos(r135), math.sin(r135)], [0, -math.sin(r135), math.cos(r135)]], dtype=torch.float64
    )
    origin_drb_gpu = torch.tensor(origin_drb, device=device, dtype=torch.float32)
    
    # (!!!) (新增) (GPU 加速)
    ZYQ = ZYQ_cpu.to(device=device, dtype=torch.float64)
    ZYQ_1 = ZYQ_1_cpu.to(device=device, dtype=torch.float64)


    for i, rgb_name in enumerate(tqdm(images_name_sorted, desc="投影点云")):
        if i < hparams.start or i >= hparams.end:
            continue
            
        # (!!!) (修复) 如果索引超出 .npy 文件数，则停止
        if i >= num_segments:
            continue

        # (图像加载和畸变校正逻辑保持不变)
        img1 = cv2.imread(str(original_images_path) + '/' + original_image_name_sorted[i])
        if img1 is None:
            tqdm.write(f"警告: 无法读取图像 {original_image_name_sorted[i]}，跳过帧 {i}")
            continue
        img_change = cv2.undistort(img1, camera_matrix1, distortion_coeffs1, None, camera_matrix_ori)
        img_change = cv2.resize(img_change, (
            int(img_change.shape[1] / hparams.down_scale), int(img_change.shape[0] / hparams.down_scale)))
        
        current_i = i # 当前帧索引

        if hparams.debug:
            cv2.imwrite(str(output_path / 'debug' / f'{i:06d}_1_rgbs.png'), img_change)

        # ==============================================================================
        # --- (!!!) (性能优化) 按需加载 .npy 文件 ---
        if current_i >= num_segments:
            tqdm.write(f"警告: 索引 {current_i} 超出点云分段数 ({num_segments})。跳过。")
            continue
        try:
            segment_path_current = segment_dir / f'{current_i:06d}.npy'
            points_data_current = np.load(segment_path_current)
        except FileNotFoundError:
            tqdm.write(f"警告: 找不到文件 {segment_path_current}。跳过帧 {i}")
            continue
        points_nerf_list = [points_data_current[:, :3].copy()]
        points_color_list = [points_data_current[:, 3:].copy()] if hparams.debug else None
        if current_i - 1 >= 0:
            try:
                segment_path_prev = segment_dir / f'{current_i - 1:06d}.npy'
                points_data_prev = np.load(segment_path_prev)
                points_nerf_list.append(points_data_prev[:, :3])
                if hparams.debug:
                    points_color_list.append(points_data_prev[:, 3:])
            except FileNotFoundError:
                pass # 静默失败
        if current_i + 1 < num_segments:
            try:
                segment_path_next = segment_dir / f'{current_i + 1:06d}.npy'
                points_data_next = np.load(segment_path_next)
                points_nerf_list.append(points_data_next[:, :3])
                if hparams.debug:
                    points_color_list.append(points_data_next[:, 3:])
            except FileNotFoundError:
                pass # 静默失败

        points_nerf_cpu = np.concatenate(points_nerf_list, axis=0)
        if hparams.debug:
            points_color = np.concatenate(points_color_list, axis=0)
        else:
            points_color = None
        # --- (性能优化) 结束 ---
        # ==============================================================================
        
        total_points_matched = len(points_nerf_cpu)
        if total_points_matched == 0:
            log_line = f"Image {i:06d}: Matched=0 | Projected=0"
            log_file.write(log_line + "\n")
            if hparams.debug or i == hparams.start:
                tqdm.write(log_line)
            continue

        # (!!!) (新增) (GPU 加速) 坐标变换
        points_nerf_gpu = torch.tensor(points_nerf_cpu, device=device, dtype=torch.float64)
        points_nerf_gpu = ZYQ @ points_nerf_gpu.T
        points_nerf_gpu = (ZYQ_1 @ points_nerf_gpu).T
        points_nerf_gpu = points_nerf_gpu.to(dtype=torch.float32)
        points_nerf_gpu = (points_nerf_gpu - origin_drb_gpu) / pose_scale_factor

        # (加载相机位姿逻辑保持不变...)
        metadata_path = metadata_dir / f'{i:06d}.pt'
        if not metadata_path.exists():
            tqdm.write(f"警告: 元数据文件 {metadata_path} 未找到，跳过帧 {i}")
            continue
        
        metadata = torch.load(metadata_path, map_location='cpu')

        # (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
        # --- (核心修复) 恢复正确的 w2c (4x4 矩阵) 计算逻辑 ---
        # (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!) (!!!)
        
        # 1. 加载 (3, 4) c2w 矩阵
        c2w_3x4_cpu = metadata['c2w'].to(dtype=torch.float32) # (3, 4)

        # 2. 复制您在原始脚本 2 中的 'E2' 变换
        #    (这假设 c2w_3x4_cpu 已经是应用了 ZYQ 变换的)
        E2 = c2w_3x4_cpu.numpy() # (3, 4)
        
        # 3. 应用您原始的 stack 变换
        E2_transformed = np.stack([
            E2[:, 0], 
            E2[:, 1] * -1, 
            E2[:, 2] * -1, 
            E2[:, 3]
        ], axis=1) # (3, 4)
        
        # 4. 添加 [0, 0, 0, 1] 行使其变为 (4, 4)
        E2_4x4 = np.concatenate((E2_transformed, [[0, 0, 0, 1]]), axis=0) # (4, 4)
        
        # 5. (安全地) 对 (4, 4) 矩阵求逆
        try:
            w2c_np = np.linalg.inv(E2_4x4) # (4, 4)
        except np.linalg.LinAlgError as e:
            tqdm.write(f"警告: 帧 {i} 的 c2w 矩阵求逆失败: {e}。跳过此帧。")
            continue
            
        # 6. 加载内参
        camera_matrix_np = np.array([[metadata['intrinsics'][0] / down_scale, 0, metadata['intrinsics'][2] / down_scale],
                                     [0, metadata['intrinsics'][1] / down_scale, metadata['intrinsics'][3] / down_scale],
                                     [0, 0, 1]], dtype=np.float32)
        
        # 7. 将最终的矩阵发送到 GPU
        # w2c_gpu = torch.tensor(w2c_np, device=device)
        w2c_gpu = torch.tensor(w2c_np, device=device, dtype=torch.float32)
        camera_matrix_gpu = torch.tensor(camera_matrix_np, device=device)
        
        # --- (!!!) (核心修复结束) (!!!) ---

        # (!!!) (新增) (GPU 加速) 投影点
        points_homogeneous_gpu = torch.cat(
            (points_nerf_gpu, torch.ones(total_points_matched, 1, device=device, dtype=torch.float32)), dim=1
        )
        pt_3d_trans_gpu = torch.matmul(w2c_gpu, points_homogeneous_gpu.T)
        pt_2d_trans_gpu = torch.matmul(camera_matrix_gpu, pt_3d_trans_gpu[:3])
        
        z = pt_2d_trans_gpu[2]
        z[z == 0] = 1e-6
        pt_2d_trans_gpu = pt_2d_trans_gpu / z
        
        projected_points_gpu = pt_2d_trans_gpu[:2]

        # (!!!) (新增) (GPU 加速) 过滤点
        image_width, image_height = int(5472 / down_scale), int(3648 / down_scale)
        mask_x_gpu = (projected_points_gpu[0, :] >= 0) & (projected_points_gpu[0, :] < image_width)
        mask_y_gpu = (projected_points_gpu[1, :] >= 0) & (projected_points_gpu[1, :] < image_height)
        mask_z_gpu = pt_3d_trans_gpu[2, :] > 0
        mask_gpu = mask_x_gpu & mask_y_gpu & mask_z_gpu
        
        effective_points_projected = torch.sum(mask_gpu).item()
        log_line = f"Image {i:06d}: Matched={total_points_matched:<10} | Projected={effective_points_projected:<10}"
        log_file.write(log_line + "\n")
        if hparams.debug or i == hparams.start:
            tqdm.write(log_line)
            
        if effective_points_projected == 0:
            if hparams.debug == False:
                 depth_map = 1e6 * np.ones((image_height, image_width, 1), dtype=np.float32)
                 np.save(depth_dir / f'{i:06d}.npy', depth_map)
            continue

        # (!!!) (新增) (GPU 加速) 将*过滤后*的点移回 CPU
        projected_points = projected_points_gpu[:, mask_gpu].cpu().numpy()
        depth_z = pt_3d_trans_gpu[2, mask_gpu].cpu().numpy()

        # (!!!) (新增) (矢量化) 使用 lexsort 填充深度图
        large_int = 1e6
        depth_map = large_int * np.ones((image_height, image_width, 1), dtype=np.float32)
        ix = projected_points[0].astype(np.int64)
        iy = projected_points[1].astype(np.int64)
        indices = np.lexsort((depth_z, ix, iy))
        sorted_ix = ix[indices]
        sorted_iy = iy[indices]
        sorted_depths = depth_z[indices]
        mask_unique = np.ones_like(sorted_ix, dtype=bool)
        mask_unique[1:] = (sorted_ix[1:] != sorted_ix[:-1]) | (sorted_iy[1:] != sorted_iy[:-1])
        final_ix = sorted_ix[mask_unique]
        final_iy = sorted_iy[mask_unique]
        final_depths = sorted_depths[mask_unique]
        depth_map[final_iy, final_ix] = final_depths[:, np.newaxis]
        
        # (!!!) (矢量化) 移除了缓慢的 'for' 循环

        # (保存或可视化结果 - 逻辑保持不变)
        if hparams.debug == False:
            np.save(depth_dir / f'{i:06d}.npy', depth_map)
        else:
            # (Debug 可视化逻辑)
            points_color_mask = points_color[mask_gpu.cpu().numpy()]
            image = 255 * np.ones((image_height, image_width, 3), dtype=np.uint8)
            
            # (!!!) (新增) (矢量化) 颜色填充
            sorted_colors = points_color_mask[indices]
            final_colors = sorted_colors[mask_unique]
            image[final_iy, final_ix] = final_colors[:, ::-1] # BGR -> RGB (或 RGB -> BGR)

            # (保存 debug 图像)
            Image.fromarray(image).save(str(output_path / 'debug' / f'{i:06d}_2_project.png'))
            
            # (深度图可视化逻辑保持不变)
            invalid_mask = (depth_map == large_int)
            depth_map_valid = depth_map[~invalid_mask]
            if depth_map_valid.size > 0:
                min_depth, max_depth = depth_map_valid.min(), depth_map_valid.max()
                depth_vis = depth_map.copy()
                depth_vis[invalid_mask] = min_depth
                depth_vis = (depth_vis - min_depth) / max(max_depth - min_depth, 1e-8)
                depth_vis = ((1 - depth_vis) * 255).astype(np.uint8)
                depth_vis_color = cv2.applyColorMap(depth_vis.squeeze(), cv2.COLORMAP_JET)
                depth_vis_color[invalid_mask.squeeze()] = [255, 255, 255]
                Image.fromarray(depth_vis_color).save(str(output_path / 'debug' / f'{i:06d}_3_depth_filter.png'))

    # (脚本结束逻辑保持不变)
    log_file.close()
    print("\n" + "=" * 25 + " 脚本执行完毕 " + "=" * 25)
    print(f"投影日志已保存到: {log_file_path}")


if __name__ == '__main__':
    main(_get_opts())
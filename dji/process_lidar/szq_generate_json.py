# # import os
# # import json

# # def process_datasets():
# #     """
# #     处理所有数据集，读取split文件，并生成 train.json 和 test.json。
# #     """
    
# #     # 1. 定义数据集信息
# #     # 结构：[ (名称, 图像目录, 深度目录, split文件, cam_in [fx, fy, cx, cy]), ... ]
    
# #     base_path = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT"
    
# #     datasets_info = [
# #         (
# #             "hav",
# #             os.path.join(base_path, "images_gt_hav/images_downsampled"),
# #             os.path.join(base_path, "depth_gt_hav"),
# #             os.path.join(base_path, "splits_hav.txt"),
# #             [3698.90185546875, 3698.90185546875, 2765.1443261913664, 1784.9986677366905] # fx, fy, cx, cy
# #         ),
# #         (
# #             "SMBU",
# #             os.path.join(base_path, "images_gt_SMBU/images_downsampled"),
# #             os.path.join(base_path, "depth_gt_SMBU"),
# #             os.path.join(base_path, "splits_SMBU.txt"),
# #             [3698.96240234375, 3698.96240234375, 2764.9719793422264, 1786.0836901728107] # fx, fy, cx, cy
# #         ),
# #         (
# #             "sztu",
# #             os.path.join(base_path, "images_gt_sztu/images_downsampled"),
# #             os.path.join(base_path, "depth_gt_sztu"),
# #             os.path.join(base_path, "splits_sztu.txt"),
# #             [3700.712158203125, 3700.712158203125, 2765.3523833447398, 1785.7838288597704] # fx, fy, cx, cy
# #         )
# #     ]
    
# #     # 2. 初始化输出列表
# #     train_data = []
# #     test_data = []
    
# #     total_train_count = 0
# #     total_test_count = 0

# #     # 3. 遍历并处理每个数据集
# #     for name, img_dir, depth_dir, split_file, cam_in in datasets_info:
# #         print(f"--- Processing dataset: {name} ---")
        
# #         current_train_count = 0
# #         current_test_count = 0
        
# #         try:
# #             with open(split_file, 'r') as f:
# #                 lines = f.readlines()
                
# #             # 跳过表头
# #             for line in lines[1:]: 
# #                 line = line.strip()
# #                 if not line:
# #                     continue
                
# #                 parts = line.split('-')
                
# #                 if len(parts) < 4:
# #                     print(f"Skipping malformed line: {line}")
# #                     continue
                    
# #                 img_name = parts[0]         # e.g., "000053.jpg"
# #                 split_type = parts[-1]      # e.g., "test"
                
# #                 # *** 关键假设：根据 000053.jpg 推断 000053.npy ***
# #                 base_name = os.path.splitext(img_name)[0]
# #                 depth_name = f"{base_name}.npy"
                
# #                 # 构建绝对路径
# #                 rgb_path = os.path.join(img_dir, img_name)
# #                 depth_path = os.path.join(depth_dir, depth_name)
                
# #                 # 创建数据条目
# #                 entry = {
# #                     "rgb": rgb_path,
# #                     "depth": depth_path,
# #                     "cam_in": cam_in
# #                 }
                
# #                 # 4. 添加到对应的列表
# #                 if split_type == "train":
# #                     train_data.append(entry)
# #                     current_train_count += 1
# #                 elif split_type == "test":
# #                     test_data.append(entry)
# #                     current_test_count += 1
            
# #             print(f"Found {current_train_count} train samples.")
# #             print(f"Found {current_test_count} test samples.")
# #             total_train_count += current_train_count
# #             total_test_count += current_test_count

# #         except FileNotFoundError:
# #             print(f"Error: Split file not found at {split_file}")
# #         except Exception as e:
# #             print(f"An error occurred while processing {name}: {e}")

# #     # 5. 将结果写入 JSON 文件
# #     output_dir = os.getcwd() # 将 JSON 文件保存在当前运行脚本的目录
# #     train_json_path = os.path.join(output_dir, "train.json")
# #     test_json_path = os.path.join(output_dir, "test.json")

# #     try:
# #         with open(train_json_path, 'w') as f:
# #             json.dump(train_data, f, indent=4)
# #         print(f"\nSuccessfully wrote {total_train_count} entries to {train_json_path}")

# #         with open(test_json_path, 'w') as f:
# #             json.dump(test_data, f, indent=4)
# #         print(f"Successfully wrote {total_test_count} entries to {test_json_path}")
        
# #     except IOError as e:
# #         print(f"Error writing JSON files: {e}")

# # # --- 运行主函数 ---
# # if __name__ == "__main__":
# #     process_datasets()

# import os
# import json

# def process_datasets():
#     """
#     处理所有数据集，读取split文件，并生成 train.json 和 test.json。
#     使用 4 倍下采样后的相机内参。
#     """
    
#     # 1. 定义数据集信息
#     # 结构：[ (名称, 图像目录, 深度目录, split文件, cam_in [fx, fy, cx, cy]), ... ]
    
#     base_path = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT"
    
#     # --- *** 使用 4 倍下采样后的新内参 *** ---
#     datasets_info = [
#         (
#             "hav",
#             os.path.join(base_path, "images_gt_hav/images_downsampled"),
#             os.path.join(base_path, "depth_gt_hav"),
#             os.path.join(base_path, "splits_hav.txt"),
#             [924.7254638671875, 924.7254638671875, 691.2860815478416, 446.2496669341726] # 4x Downsampled
#         ),
#         (
#             "SMBU",
#             os.path.join(base_path, "images_gt_SMBU/images_downsampled"),
#             os.path.join(base_path, "depth_gt_SMBU"),
#             os.path.join(base_path, "splits_SMBU.txt"),
#             [924.7406005859375, 924.7406005859375, 691.2429948355566, 446.5209225432027] # 4x Downsampled
#         ),
#         (
#             "sztu",
#             os.path.join(base_path, "images_gt_sztu/images_downsampled"),
#             os.path.join(base_path, "depth_gt_sztu"),
#             os.path.join(base_path, "splits_sztu.txt"),
#             [925.1780395507812, 925.1780395507812, 691.3380958361849, 446.4459572149426] # 4x Downsampled
#         )
#     ]
    
#     # 2. 初始化输出列表
#     train_data = []
#     test_data = []
    
#     total_train_count = 0
#     total_test_count = 0

#     # 3. 遍历并处理每个数据集
#     for name, img_dir, depth_dir, split_file, cam_in in datasets_info:
#         print(f"--- Processing dataset: {name} ---")
        
#         current_train_count = 0
#         current_test_count = 0
        
#         try:
#             with open(split_file, 'r') as f:
#                 lines = f.readlines()
                
#             # 跳过表头
#             for line in lines[1:]: 
#                 line = line.strip()
#                 if not line:
#                     continue
                
#                 parts = line.split('-')
                
#                 if len(parts) < 4:
#                     print(f"Skipping malformed line: {line}")
#                     continue
                    
#                 img_name = parts[0]         # e.g., "000053.jpg"
#                 split_type = parts[-1]      # e.g., "test"
                
#                 # *** 关键假设：根据 000053.jpg 推断 000053.npy ***
#                 base_name = os.path.splitext(img_name)[0]
#                 depth_name = f"{base_name}.npy"
                
#                 # 构建绝对路径
#                 rgb_path = os.path.join(img_dir, img_name)
#                 depth_path = os.path.join(depth_dir, depth_name)
                
#                 # 创建数据条目
#                 entry = {
#                     "rgb": rgb_path,
#                     "depth": depth_path,
#                     "cam_in": cam_in
#                 }
                
#                 # 4. 添加到对应的列表
#                 if split_type == "train":
#                     train_data.append(entry)
#                     current_train_count += 1
#                 elif split_type == "test":
#                     test_data.append(entry)
#                     current_test_count += 1
            
#             print(f"Found {current_train_count} train samples.")
#             print(f"Found {current_test_count} test samples.")
#             total_train_count += current_train_count
#             total_test_count += current_test_count

#         except FileNotFoundError:
#             print(f"Error: Split file not found at {split_file}")
#         except Exception as e:
#             print(f"An error occurred while processing {name}: {e}")

#     # 5. 将结果写入 JSON 文件
#     output_dir = os.getcwd() # 将 JSON 文件保存在当前运行脚本的目录
#     train_json_path = os.path.join(output_dir, "train.json")
#     test_json_path = os.path.join(output_dir, "test.json")

#     try:
#         with open(train_json_path, 'w') as f:
#             json.dump(train_data, f, indent=4)
#         print(f"\nSuccessfully wrote {total_train_count} entries to {train_json_path}")

#         with open(test_json_path, 'w') as f:
#             json.dump(test_data, f, indent=4)
#         print(f"Successfully wrote {total_test_count} entries to {test_json_path}")
        
#     except IOError as e:
#         print(f"Error writing JSON files: {e}")

# # --- 运行主函数 ---
# if __name__ == "__main__":
#     process_datasets()


import os
import json

def process_datasets():
    """
    处理所有数据集，读取split文件，并生成 train.json 和 test.json。
    使用 4 倍下采样后的相机内参。
    输出格式为 {"files": [...] }。
    """
    
    # 1. 定义数据集信息
    # 结构：[ (名称, 图像目录, 深度目录, split文件, cam_in [fx, fy, cx, cy]), ... ]
    
    base_path = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT"
    
    # --- 使用 4 倍下采样后的新内参 ---
    datasets_info = [
        (
            "hav",
            os.path.join(base_path, "images_gt_hav/images_downsampled"),
            os.path.join(base_path, "depth_gt_hav"),
            os.path.join(base_path, "splits_hav.txt"),
            [924.7254638671875, 924.7254638671875, 691.2860815478416, 446.2496669341726] # 4x Downsampled
        ),
        (
            "SMBU",
            os.path.join(base_path, "images_gt_SMBU/images_downsampled"),
            os.path.join(base_path, "depth_gt_SMBU"),
            os.path.join(base_path, "splits_SMBU.txt"),
            [924.7406005859375, 924.7406005859375, 691.2429948355566, 446.5209225432027] # 4x Downsampled
        ),
        (
            "sztu",
            os.path.join(base_path, "images_gt_sztu/images_downsampled"),
            os.path.join(base_path, "depth_gt_sztu"),
            os.path.join(base_path, "splits_sztu.txt"),
            [925.1780395507812, 925.1780395507812, 691.3380958361849, 446.4459572149426] # 4x Downsampled
        ),
        (
            "lfls2",
            os.path.join(base_path, "images_gt_lfls2/images_downsampled"),
            os.path.join(base_path, "depth_gt_lfls2"),
            os.path.join(base_path, "splits_lfls2.txt"),
            [922.785, 922.785, 688.9025, 449.135] # 4x Downsampled
        ),
        (
            "upper",
            os.path.join(base_path, "images_gt_upper/images_downsampled"),
            os.path.join(base_path, "depth_gt_upper"),
            os.path.join(base_path, "splits_upper.txt"),
            [922.5975, 922.5975, 688.9425,449.02] # 4x Downsampled
        )
    ]
    
    # 2. 初始化输出列表
    train_data_list = []
    test_data_list = []
    
    total_train_count = 0
    total_test_count = 0

    # 3. 遍历并处理每个数据集
    for name, img_dir, depth_dir, split_file, cam_in in datasets_info:
        print(f"--- Processing dataset: {name} ---")
        
        current_train_count = 0
        current_test_count = 0
        
        try:
            with open(split_file, 'r') as f:
                lines = f.readlines()
                
            # 跳过表头
            for line in lines[1:]: 
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('-')
                
                if len(parts) < 4:
                    print(f"Skipping malformed line: {line}")
                    continue
                    
                img_name = parts[0]         # e.g., "000053.jpg"
                split_type = parts[-1]      # e.g., "test"
                
                # *** 关键假设：根据 000053.jpg 推断 000053.npy ***
                base_name = os.path.splitext(img_name)[0]
                depth_name = f"{base_name}.npy"
                
                # 构建绝对路径
                rgb_path = os.path.join(img_dir, img_name)
                depth_path = os.path.join(depth_dir, depth_name)
                
                # 创建数据条目
                entry = {
                    "rgb": rgb_path,
                    "depth": depth_path,
                    "cam_in": cam_in
                }
                
                # 4. 添加到对应的列表
                if split_type == "train":
                    train_data_list.append(entry)
                    current_train_count += 1
                elif split_type == "test":
                    test_data_list.append(entry)
                    current_test_count += 1
            
            print(f"Found {current_train_count} train samples.")
            print(f"Found {current_test_count} test samples.")
            total_train_count += current_train_count
            total_test_count += current_test_count

        except FileNotFoundError:
            print(f"Error: Split file not found at {split_file}")
        except Exception as e:
            print(f"An error occurred while processing {name}: {e}")

    # 5. 将结果写入 JSON 文件
    output_dir = os.getcwd() # 将 JSON 文件保存在当前运行脚本的目录
    train_json_path = os.path.join(output_dir, "train.json")
    test_json_path = os.path.join(output_dir, "test.json")

    # *** 关键更改：将列表包装在 "files" 键中 ***
    train_output_dict = {"files": train_data_list}
    test_output_dict = {"files": test_data_list}

    try:
        # 写入 train.json
        with open(train_json_path, 'w') as f:
            json.dump(train_output_dict, f, indent=4)
        print(f"\nSuccessfully wrote {total_train_count} entries to {train_json_path}")

        # 写入 test.json
        with open(test_json_path, 'w') as f:
            json.dump(test_output_dict, f, indent=4)
        print(f"Successfully wrote {total_test_count} entries to {test_json_path}")
        
    except IOError as e:
        print(f"Error writing JSON files: {e}")

# --- 运行主函数 ---
if __name__ == "__main__":
    process_datasets()


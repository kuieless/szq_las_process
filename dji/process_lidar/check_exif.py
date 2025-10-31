# import sys
# from PIL import Image
# from PIL.ExifTags import TAGS, GPSTAGS

# def check_exif(image_path):
#     print(f"--- 正在检查文件: {image_path} ---")
    
#     try:
#         img = Image.open(image_path)
#     except Exception as e:
#         print(f"!!! 错误: 无法打开图片: {e}")
#         return

#     # 1. 检查 EXIF 数据
#     try:
#         exif_data = img._getexif()
#     except Exception as e:
#         print(f"!!! 错误: 读取 EXIF 时发生错误: {e}")
#         return

#     if not exif_data:
#         print("!!! 结果: 未找到 EXIF 元数据。")
        
#         # 尝试检查 XMP 元数据 (Pillow 对 XMP 的支持有限, 但值得一试)
#         xmp_data = img.getxmp()
#         if xmp_data:
#             print("--- 补充: 找到了 XMP 元数据 (内容未解析) ---")
#         return
        
#     print(f"\n[1] 成功读取 EXIF. 共 {len(exif_data)} 个标签。")

#     # 2. 检查 GPSInfo 标签 (ID: 34853)
#     if 34853 not in exif_data:
#         print("!!! 结果: 找到了 EXIF，但未找到 GPSInfo 标签 (ID 34853)。")
#         print("\n--- 找到的 EXIF 标签 (部分) ---")
#         # 打印前 20 个标签，帮助我们找到线索
#         count = 0
#         for tag_id, value in exif_data.items():
#             if count >= 20:
#                 break
#             tag_name = TAGS.get(tag_id, tag_id)
#             print(f"  Tag: {tag_name} (ID: {tag_id})")
#             count += 1
#         return

#     print("\n[2] 成功找到 GPSInfo 标签 (ID 34853)。")

#     # 3. 解析 GPSInfo 内部
#     gps_info = {}
#     try:
#         for tag, value in exif_data[34853].items():
#             tag_name = GPSTAGS.get(tag, tag)
#             gps_info[tag_name] = value
#     except Exception as e:
#         print(f"!!! 错误: 解析 GPSInfo 内部时出错: {e}")
#         return

#     print("\n[3] GPSInfo 内部所有标签:")
#     if not gps_info:
#         print("  (GPSInfo 块为空)")
#         return
        
#     for tag_name, value in gps_info.items():
#         # GPSTAGS 是一个反向查找的字典, 我们需要找到原始 ID
#         raw_id = "N/A"
#         for k, v in GPSTAGS.items():
#             if v == tag_name:
#                 raw_id = k
#                 break
#         print(f"  - {tag_name} (ID: {raw_id}): {value}")

#     # 4. 检查关键的经纬度标签
#     lat_dms = gps_info.get('GPSLatitude')
#     lat_ref = gps_info.get('GPSLatitudeRef')
#     lon_dms = gps_info.get('GPSLongitude')
#     lon_ref = gps_info.get('GPSLongitudeRef')

#     print("\n[4] 检查关键标签:")
#     print(f"  - GPSLatitude (DMS): {lat_dms}")
#     print(f"  - GPSLatitudeRef (N/S): {lat_ref}")
#     print(f"  - GPSLongitude (DMS): {lon_dms}")
#     print(f"  - GPSLongitudeRef (E/W): {lon_ref}")

#     if lat_dms and lat_ref and lon_dms and lon_ref:
#         print("\n--- 诊断结果: 成功 ---")
#         print("所有关键的 GPS 标签都已找到。主脚本理论上应该可以工作。")
#     else:
#         print("\n--- 诊断结果: 失败 ---")
#         print("!!! 错误: 缺少一个或多个关键的 GPS 标签 (Latitude, Longitude, or Ref)。")

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("用法: python check_exif.py [你的图片文件路径]")
#     else:
#         check_exif(sys.argv[1])
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

# --- 你需要修改的部分 ---
# 1. 指定你的 .pt 元数据文件 (请使用完整路径)
file_to_check = r'/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/hav/metadata/000000.pt'  # <-- 修改这里
# -------------------------

print(f"--- 正在检查文件: {file_to_check} ---")

# 检查文件是否存在
if not os.path.exists(file_to_check):
    print("\n错误：文件未找到！请确认路径和文件名是否正确。")
else:
    try:
        # 加载 .pt 文件
        # 使用 map_location='cpu' 是一个好习惯，
        # 这样即使文件是在 GPU 上保存的，在没有 GPU 的机器上也能打开
        data = torch.load(file_to_check, map_location='cpu')

        print(f"\n[文件加载成功]")
        data_type = type(data)
        print(f"顶层数据类型 (Top-level Type): {data_type}")

        # --- 1. 如果数据是字典 (最常见，例如元数据或 model state_dict) ---
        if isinstance(data, dict):
            print(f"字典中的项目数量 (Keys): {len(data)}")
            print("\n[字典内容预览 (前 10 项)]: ")
            
            for i, (key, value) in enumerate(data.items()):
                if i >= 10:
                    print("... (还有更多项目)")
                    break
                
                print(f"\n  - 键 (Key): '{key}'")
                value_type = type(value)
                print(f"    值类型 (Value Type): {value_type}")

                # 如果值是 Tensor，打印详细信息
                if isinstance(value, torch.Tensor):
                    print(f"    Tensor 形状 (Shape): {value.shape}")
                    print(f"    Tensor 类型 (dtype): {value.dtype}")
                    print(f"    Tensor 设备 (Device): {value.device}")
                # 如果值是列表/元组，打印长度
                elif isinstance(value, (list, tuple)):
                    print(f"    列表/元组 长度 (Length): {len(value)}")
                # 如果是其他类型，打印一个简短的预览
                else:
                    try:
                        value_str = str(value)
                        if len(value_str) > 70:
                            value_str = value_str[:70] + "..."
                        print(f"    值 (Value) 预览: {value_str}")
                    except Exception:
                        print("    值 (Value): [无法显示预览]")

        # --- 2. 如果数据是 Tensor (类似于你的 .npy 例子) ---
        elif isinstance(data, torch.Tensor):
            print(f"\n[基本信息]")
            print(f"数据类型 (dtype): {data.dtype}")
            print(f"数据形状 (Shape): {data.shape}")
            print(f"数据设备 (Device): {data.device}")
            print(f"维度 (Dimensions): {data.dim()}")

            # --- 分析数据内容 ---
            print(f"\n[数据分析]")
            # 转换为 float 进行统计，以避免整数溢出或类型问题
            try:
                # 排除 NaN 和 Inf
                valid_data = data[torch.isfinite(data.float())]
                
                if valid_data.numel() > 0:
                    print(f"有效 (finite) 数据点数量: {valid_data.numel()}")
                    # .item() 用于从 0 维张量中获取 Python 数字
                    print(f"最小值 (Min): {valid_data.min().item():.4f}")
                    print(f"最大值 (Max): {valid_data.max().item():.4f}")
                    print(f"平均值 (Mean): {valid_data.mean().item():.4f}")
                else:
                    print("警告：Tensor 中没有找到有效的 (finite) 数据点！")
                
                # --- 可视化 (如果维度合适) ---
                if data.dim() == 2: # 2D 张量，像深度图
                    print("\n正在生成 Tensor 可视化... (如果看到一个窗口，请关闭它以继续)")
                    plt.imshow(data.numpy(), cmap='jet') # 转换为 numpy 进行绘图
                    plt.title(f'Tensor Preview for {os.path.basename(file_to_check)}')
                    plt.colorbar(label='Value')
                    plt.show()
                elif data.dim() == 3 and data.shape[0] in [1, 3]: # 可能是 (1, H, W) 或 (3, H, W) 图像
                    print("\n检测到 3D 图像类 Tensor，尝试可视化...")
                    # 转换 (C, H, W) -> (H, W, C) 以便 plt.imshow
                    img_data = data.permute(1, 2, 0).numpy().squeeze()
                    plt.imshow(img_data, cmap='gray' if img_data.ndim == 2 else None)
                    plt.title(f'Tensor Preview for {os.path.basename(file_to_check)}')
                    plt.show()
                else:
                    print("\nTensor 维度不适合 2D 可视化 (非 2D 或 3D 图像格式)。")

            except Exception as e:
                print(f"Tensor 数据分析或可视化时出错: {e}")

        # --- 3. 如果数据是列表或元组 ---
        elif isinstance(data, (list, tuple)):
            print(f"列表/元组 中的项目数量: {len(data)}")
            if len(data) > 0:
                print("\n[内容预览 (前 5 项)]: ")
                for i, item in enumerate(data[:5]):
                    print(f"  - 项目 {i} (类型: {type(item)}):")
                    if isinstance(item, torch.Tensor):
                        print(f"    Tensor 形状 (Shape): {item.shape}, 类型 (dtype): {item.dtype}")
                    else:
                        try:
                            item_str = str(item)
                            if len(item_str) > 70:
                                item_str = item_str[:70] + "..."
                            print(f"    值 (Value) 预览: {item_str}")
                        except Exception:
                            print("    值 (Value): [无法显示预览]")
                if len(data) > 5:
                    print("... (还有更多项目)")

        # --- 4. 其他数据类型 (例如 int, float, str, 或一个完整的模型对象) ---
        else:
            print("\n[数据预览]:")
            try:
                data_str = str(data)
                if len(data_str) > 500: # 完整模型 str() 可能会非常长
                    print(data_str[:500] + "\n... (内容过长，已截断)")
                else:
                    print(data)
            except Exception:
                print("[无法显示数据预览]")

        print("\n--- 检查完成 ---")

    except Exception as e:
        print(f"\n读取或解析文件时发生错误: {e}")
        print("提示：这可能不是一个有效的 .pt 文件，或者文件已损坏，或者它是由 torch.save(model) 保存的完整模型，而不是元数据。")
#         #python check_exif.py /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/hav/image_metadata/000000.pt
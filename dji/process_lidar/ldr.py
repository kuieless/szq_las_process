# import numpy as np
# import laspy
#
# # --- 1. 定义你的 .las 文件路径 ---
# # 请将这里的路径替换为你自己的 .las 文件路径
# # las_file_path = 'F:\download\SMBU2\lidars\\terra_las\cloud_merged.las'
# las_file_path = 'F:\download\SMBU2\SMBU1\lidars\\terra_las\cloud_merged.las'
#
# try:
#     # --- 2. 打开并读取 .las 文件 ---
#     # 使用 laspy.read() 是最简单的方式，它会将所有数据加载到内存中
#     # 对于非常大的文件，可以使用 laspy.open() 进行流式读取
#     print(f"正在读取文件: {las_file_path}\n")
#     las = laspy.read(las_file_path)
#
#     # --- 3. 查看文件的头部信息 (Header) ---
#     # 头部信息包含了文件的元数据，非常重要
#     print("--- 文件头部信息 (Metadata) ---")
#     header = las.header
#     print(f"LAS 版本: {header.version}")
#     print(f"点数据格式 ID: {header.point_format.id}")
#     print(f"点数量: {header.point_count}")
#
#     # 坐标范围 (min/max)
#     print("\n坐标范围 (Bounding Box):")
#     print(f"  X 范围: {header.x_min:.2f} - {header.x_max:.2f}")
#     print(f"  Y 范围: {header.y_min:.2f} - {header.y_max:.2f}")
#     print(f"  Z 范围: {header.z_min:.2f} - {header.z_max:.2f}")
#
#     # 坐标缩放与偏移量 (用于将存储的整数值还原为真实的坐标)
#     # 真实坐标 = (存储值 * 缩放系数) + 偏移量
#     print("\n坐标缩放与偏移:")
#     print(f"  X scale: {header.x_scale}, X offset: {header.x_offset}")
#     print(f"  Y scale: {header.y_scale}, Y offset: {header.y_offset}")
#     print(f"  Z scale: {header.z_scale}, Z offset: {header.z_offset}")
#
#     # --- 4. 查看点数据的维度 (Dimensions) ---
#     # 也就是每个点都记录了哪些信息
#     print("\n--- 点数据包含的维度 ---")
#     point_format = las.point_format
#     # point_format.dimension_names 会返回一个包含所有维度名称的列表
#     print(f"所有维度: {list(point_format.dimension_names)}\n")
#
#     # --- 5. 读取并查看具体的点数据 ---
#     # las 对象本身就像一个 NumPy 数组，我们可以直接访问点的信息
#     print("--- 查看前 5 个点的数据 ---")
#     # 如果点数量小于5，则只显示存在的点
#     num_points_to_show = min(5, len(las.points))
#
#     if num_points_to_show > 0:
#         for i in range(num_points_to_show):
#             # 获取真实的坐标值
#             x = las.X[i]
#             y = las.Y[i]
#             z = las.Z[i]
#
#             # 获取其他常见属性
#             intensity = las.intensity[i] if 'intensity' in point_format.dimension_names else 'N/A'
#             return_number = las.return_number[i] if 'return_number' in point_format.dimension_names else 'N/A'
#             classification = las.classification[i] if 'classification' in point_format.dimension_names else 'N/A'
#             gps_time = las.gps_time[i] if 'gps_time' in point_format.dimension_names else 'N/A'
#
#             print(f"点 {i}:")
#             print(f"  坐标 (X, Y, Z): ({x:.3f}, {y:.3f}, {z:.3f})")
#             print(f"  强度 (Intensity): {intensity}")
#             print(f"  回波次序 (Return Number): {return_number}")
#             print(f"  分类 (Classification): {classification}")
#             print(f"  GPS 时间戳: {gps_time}")
#             print("-" * 20)
#
#     # --- 6. 查看一些统计信息 ---
#     # 由于点数据是 NumPy 数组，可以很方便地进行计算
#     if len(las.points) > 0:
#         print("\n--- 数据统计信息 ---")
#         avg_intensity = np.mean(las.intensity) if 'intensity' in point_format.dimension_names else 'N/A'
#         print(f"平均反射强度: {avg_intensity:.2f}")
#
#         # 查看分类信息统计 (如果存在)
#         if 'classification' in point_format.dimension_names:
#             unique_classes, counts = np.unique(las.classification, return_counts=True)
#             print("点云分类统计:")
#             for cls, count in zip(unique_classes, counts):
#                 print(f"  类别 {cls}: {count} 个点")
#
# except FileNotFoundError:
#     print(f"错误: 文件 '{las_file_path}' 未找到。请检查文件路径是否正确。")
# except Exception as e:
#     print(f"读取文件时发生错误: {e}")
import numpy as np
import laspy
import datetime  # 导入datetime模块用于时长格式化

# --- 1. 定义你的 .las 文件路径 ---
# 请将这里的路径替换为你自己的 .las 文件路径
# las_file_path = 'F:\download\SMBU2\lidars\\terra_las\cloud_merged.las'
las_file_path = 'F:\\download\\SMBU2\\SMBU1\\lidars\\terra_las\\cloud_merged.las'
# las_file_path = 'H:\BaiduNetdiskDownload\LiDAR_Raw\SZIIT\SZTIH018\lidars\\terra_las\\cloud_merged.las'
try:
    # --- 2. 打开并读取 .las 文件 ---
    # 使用 laspy.read() 是最简单的方式，它会将所有数据加载到内存中
    # 对于非常大的文件，可以使用 laspy.open() 进行流式读取
    print(f"正在读取文件: {las_file_path}\n")
    las = laspy.read(las_file_path)

    # --- 3. 查看文件的头部信息 (Header) ---
    # 头部信息包含了文件的元数据，非常重要
    print("--- 文件头部信息 (Metadata) ---")
    header = las.header
    print(f"LAS 版本: {header.version}")
    print(f"点数据格式 ID: {header.point_format.id}")
    print(f"点数量: {header.point_count}")

    # 坐标范围 (min/max)
    print("\n坐标范围 (Bounding Box):")
    print(f"  X 范围: {header.x_min:.2f} - {header.x_max:.2f}")
    print(f"  Y 范围: {header.y_min:.2f} - {header.y_max:.2f}")
    print(f"  Z 范围: {header.z_min:.2f} - {header.z_max:.2f}")

    # 坐标缩放与偏移量 (用于将存储的整数值还原为真实的坐标)
    # 真实坐标 = (存储值 * 缩放系数) + 偏移量
    print("\n坐标缩放与偏移:")
    print(f"  X scale: {header.x_scale}, X offset: {header.x_offset}")
    print(f"  Y scale: {header.y_scale}, Y offset: {header.y_offset}")
    print(f"  Z scale: {header.z_scale}, Z offset: {header.z_offset}")

    # --- 4. 查看点数据的维度 (Dimensions) ---
    # 也就是每个点都记录了哪些信息
    print("\n--- 点数据包含的维度 ---")
    point_format = las.point_format
    # point_format.dimension_names 会返回一个包含所有维度名称的列表
    print(f"所有维度: {list(point_format.dimension_names)}\n")

    # --- 5. 读取并查看具体的点数据 ---
    # las 对象本身就像一个 NumPy 数组，我们可以直接访问点的信息
    print("--- 查看前 5 个点的数据 ---")
    # 如果点数量小于5，则只显示存在的点
    num_points_to_show = min(5, len(las.points))

    if num_points_to_show > 0:
        for i in range(num_points_to_show):
            # 获取真实的坐标值
            x = las.X[i]
            y = las.Y[i]
            z = las.Z[i]

            # 获取其他常见属性
            intensity = las.intensity[i] if 'intensity' in point_format.dimension_names else 'N/A'
            return_number = las.return_number[i] if 'return_number' in point_format.dimension_names else 'N/A'
            classification = las.classification[i] if 'classification' in point_format.dimension_names else 'N/A'
            gps_time = las.gps_time[i] if 'gps_time' in point_format.dimension_names else 'N/A'

            print(f"点 {i}:")
            print(f"  坐标 (X, Y, Z): ({x:.3f}, {y:.3f}, {z:.3f})")
            print(f"  强度 (Intensity): {intensity}")
            print(f"  回波次序 (Return Number): {return_number}")
            print(f"  分类 (Classification): {classification}")
            print(f"  GPS 时间戳: {gps_time}")
            print("-" * 20)

    # --- 6. 查看一些统计信息 ---
    # 由于点数据是 NumPy 数组，可以很方便地进行计算
    if len(las.points) > 0:
        print("\n--- 数据统计信息 ---")
        # 检查 'intensity' 维度是否存在
        if 'intensity' in point_format.dimension_names:
            avg_intensity = np.mean(las.intensity)
            print(f"平均反射强度: {avg_intensity:.2f}")
        else:
            print("平均反射强度: N/A (文件中无此维度)")

        # 查看分类信息统计 (如果存在)
        if 'classification' in point_format.dimension_names:
            unique_classes, counts = np.unique(las.classification, return_counts=True)
            print("点云分类统计:")
            for cls, count in zip(unique_classes, counts):
                print(f"  类别 {cls}: {count} 个点")

        # ===================================================================
        # --- 7. (新增) 时间戳分析 ---
        print("\n--- 时间戳分析 ---")
        if 'gps_time' not in point_format.dimension_names:
            print("文件中不包含 'gps_time' 维度，无法进行时间戳分析。")
        else:
            # --- 7a. 整体时间范围 ---
            print("\n[整体时间范围]")
            gps_times = las.gps_time
            start_time = np.min(gps_times)
            end_time = np.max(gps_times)
            duration_seconds = end_time - start_time

            # 将总秒数格式化为 H:M:S
            td = datetime.timedelta(seconds=duration_seconds)

            print(f"  起始时间戳: {start_time:.6f}")
            print(f"  结束时间戳: {end_time:.6f}")
            print(f"  总 时 长: {duration_seconds:.3f} 秒 (约为 {td})")

            # --- 7b. 数据段与中断分析 ---
            # 定义一个阈值来判断什么是“中断”。如果两个连续点的时间差超过这个值，就认为是一个中断。
            # 您可以根据需要调整这个值。例如，对于飞行数据，1-2秒的中断可能代表一次转弯或新的航线。
            GAP_THRESHOLD_SECONDS = 1.0
            print(f"\n[数据段与中断分析 (中断阈值 > {GAP_THRESHOLD_SECONDS} 秒)]")

            # 对时间戳进行排序，以确保是按时间顺序处理
            sorted_times = np.sort(gps_times)

            # 计算连续点之间的时间差
            time_diffs = np.diff(sorted_times)

            # 找到所有时间差大于阈值的位置
            gap_indices = np.where(time_diffs > GAP_THRESHOLD_SECONDS)[0]

            if len(gap_indices) == 0:
                print("  在整个文件中未发现明显的数据中断。")
            else:
                print("\n  发现的数据中断 (Gaps):")
                for i, index in enumerate(gap_indices):
                    gap_start = sorted_times[index]
                    gap_end = sorted_times[index + 1]
                    gap_duration = gap_end - gap_start
                    print(f"    中断 {i + 1}: 从 {gap_start:.3f} 到 {gap_end:.3f} (持续 {gap_duration:.3f} 秒)")

                print("\n  连续的数据采集时段 (Segments):")
                last_end_time = start_time
                segment_num = 1
                for index in gap_indices:
                    segment_end = sorted_times[index]
                    duration = segment_end - last_end_time
                    print(
                        f"    时段 {segment_num}: 从 {last_end_time:.3f} 到 {segment_end:.3f} (持续 {duration:.3f} 秒)")
                    last_end_time = sorted_times[index + 1]
                    segment_num += 1

                # 打印最后一个数据段 (从最后一个中断结束到文件结尾)
                duration = end_time - last_end_time
                print(f"    时段 {segment_num}: 从 {last_end_time:.3f} 到 {end_time:.3f} (持续 {duration:.3f} 秒)")
        # ===================================================================

except FileNotFoundError:
    print(f"错误: 文件 '{las_file_path}' 未找到。请检查文件路径是否正确。")
except Exception as e:
    print(f"读取文件时发生错误: {e}")
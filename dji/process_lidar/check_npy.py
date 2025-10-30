import numpy as np
import matplotlib.pyplot as plt
import os

# --- 你需要修改的部分 ---
# 1. 指定你的输出目录
output_path = r'F:\download\SMBU2\output'

# 2. 指定你要检查的文件名 (可以任选一个)
# 检查一个验证集文件
file_to_check = 'F:\download\SMBU2\output\\train\depth_dji\\000001.npy'
# 或者检查一个训练集文件
# file_to_check = 'train/depth_dji/000001.npy'
# -------------------------

# 使用 os.path.join 来正确地组合路径
full_path = os.path.join(output_path, file_to_check)

print(f"--- 正在检查文件: {full_path} ---")

# 检查文件是否存在
if not os.path.exists(full_path):
    print("\n错误：文件未找到！请确认路径和文件名是否正确。")
else:
    try:
        # 加载 .npy 文件
        depth_map = np.load(full_path)

        # --- 打印基本信息 ---
        print(f"\n[基本信息]")
        print(f"数据类型 (dtype): {depth_map.dtype}")
        print(f"数据形状 (Shape): {depth_map.shape} (高, 宽, 通道数)")

        # --- 分析数据内容 ---
        # 你的代码中用一个很大的数 (1e6) 来表示无效的深度值
        large_int_value = 1e6

        # 筛选出有效的深度值 (不等于那个巨大值的点)
        valid_depths = depth_map[depth_map < large_int_value]

        print(f"\n[数据分析]")
        if valid_depths.size > 0:
            print(f"有效的深度点数量: {valid_depths.size}")
            print(f"最小深度值 (Min): {valid_depths.min():.4f}")
            print(f"最大深度值 (Max): {valid_depths.max():.4f}")
            print(f"平均深度值 (Mean): {valid_depths.mean():.4f}")

            # --- 可视化深度图 (可选，但非常推荐) ---
            print("\n正在生成深度图可视化... (如果看到一个窗口，请关闭它以继续)")
            plt.imshow(depth_map.squeeze(), cmap='jet')  # 使用squeeze()移除单通道维度
            plt.title(f'Depth Map Preview for {os.path.basename(full_path)}')
            plt.colorbar(label='Depth Value')
            plt.show()

        else:
            print("警告：文件中没有找到有效的深度点！所有值都是无效值。")

        print("\n--- 检查完成 ---")

    except Exception as e:
        print(f"\n读取文件时发生错误: {e}")
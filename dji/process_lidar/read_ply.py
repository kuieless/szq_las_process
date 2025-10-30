# 脚本功能：
# 读取一个 .ply 文件，并打印出其顶点(vertex)包含的所有属性。
# 帮助您快速确认该文件是否包含可用于划分的 'timestamp' 或类似属性。

from plyfile import PlyData, PlyElement
import sys

# --- 用户配置 ---
# 请将这里替换为您想检查的 .ply 文件路径
ply_file_path = r"F:\download\SMBU2\HAV\lidars\terra_point_ply\cloud_merged.ply"
# -----------------

try:
    print(f"正在读取文件: {ply_file_path}")

    # 只读取头部信息，速度很快
    plydata = PlyData.read(ply_file_path)

    # 查找名为 'vertex' 的元素
    if 'vertex' in plydata:
        vertex_element = plydata['vertex']
        properties = vertex_element.properties

        print("\n" + "=" * 10 + " 文件包含的顶点属性 " + "=" * 10)

        # 打印所有属性的名称和数据类型
        for prop in properties:
            print(f"  - 属性名称: '{prop.name}',  数据类型: {prop.val_dtype}")

        print("=" * 45)

        # 检查是否存在可能是时间戳的属性
        has_timestamp = any(p.name in ['timestamp', 'gps_time', 'time'] for p in properties)

        if has_timestamp:
            print("\n✅ 好消息！文件中似乎包含时间戳属性。")
            print("您可以修改主脚本，用 'plyfile' 库来读取点和时间戳进行划分。")
        else:
            print("\n⚠️ 注意：文件中未找到明显的时间戳属性 (如 'timestamp', 'gps_time')。")
            print("这个 .ply 文件可能无法用于精确的时间划分。")

    else:
        print("错误: 在 .ply 文件中没有找到 'vertex' 元素。")

except FileNotFoundError:
    print(f"错误: 文件未找到 at '{ply_file_path}'")
except Exception as e:
    print(f"处理文件时发生错误: {e}")

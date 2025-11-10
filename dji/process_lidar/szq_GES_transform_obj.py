# # import sys
# # import os

# # # --- 变换参数 (根据您的截图) ---
# # # Shift (平移向量)
# # SHIFT_X = -409115.53
# # SHIFT_Y = -3950464.63
# # SHIFT_Z = 0.00

# # # Scale (缩放因子)
# # SCALE = 1.00000000
# # # -------------------------------------

# # def transform_obj(input_file, output_file):
# #     """
# #     读取一个OBJ文件，应用CloudCompare的变换，并写入新文件。
# #     变换公式: Local_Coord = (Original_Coord + Shift) * Scale
# #     """
    
# #     # 检查输入文件是否存在
# #     if not os.path.exists(input_file):
# #         print(f"错误: 输入文件未找到 '{input_file}'")
# #         return

# #     print(f"开始处理 '{input_file}'...")
    
# #     try:
# #         with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
# #             vertex_count = 0
            
# #             # 逐行读取原始OBJ文件
# #             for line in infile:
                
# #                 # 检查是否为顶点坐标行 (v x y z)
# #                 if line.startswith('v '):
# #                     parts = line.split()
# #                     try:
# #                         # parts[0] 是 'v'
# #                         # 获取原始坐标
# #                         original_x = float(parts[1])
# #                         original_y = float(parts[2])
# #                         original_z = float(parts[3])
                        
# #                         # 应用变换公式
# #                         local_x = (original_x + SHIFT_X) * SCALE
# #                         local_y = (original_y + SHIFT_Y) * SCALE
# #                         local_z = (original_z + SHIFT_Z) * SCALE
                        
# #                         # 将变换后的坐标写入新文件
# #                         # 保留高精度
# #                         outfile.write(f"v {local_x:.8f} {local_y:.8f} {local_z:.8f}\n")
# #                         vertex_count += 1
                        
# #                     except (IndexError, ValueError) as e:
# #                         print(f"警告: 跳过格式错误的顶点行: {line.strip()} ({e})")
# #                         outfile.write(line) # 如果解析失败，原样写入
                
# #                 # 对于所有其他行 (vn, vt, f, #, mtllib, o, g, s 等)
# #                 # 原封不动地复制到新文件
# #                 else:
# #                     outfile.write(line)
                    
# #             print(f"\n处理完成!")
# #             print(f"总共变换了 {vertex_count} 个顶点。")
# #             print(f"已将结果保存到: '{output_file}'")

# #     except Exception as e:
# #         print(f"发生了一个错误: {e}")
# #         sys.exit(1)

# # # --- 主程序入口 ---
# # if __name__ == "__main__":
    
# #     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# #     # !!!  请在这里修改您的输入和输出文件名  !!!
# #     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# #     INPUT_OBJ_FILE = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/Mergedmesh.obj"
# #     OUTPUT_OBJ_FILE = "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/transformed_local_mesh.obj"

    
# #     # (可选) 允许通过命令行参数传入文件名
# #     # 用法: python this_script.py <input.obj> <output.obj>
# #     if len(sys.argv) == 3:
# #         input_file = sys.argv[1]
# #         output_file = sys.argv[2]
# #         print("使用命令行参数指定文件。")
# #         transform_obj(input_file, output_file)
# #     else:
# #         print(f"使用脚本中定义的默认文件名。")
# #         print(f"输入: {INPUT_OBJ_FILE}")
# #         print(f"输出: {OUTPUT_OBJ_FILE}")
        
# #         # 简单检查，防止用户未修改默认值就运行
# #         if INPUT_OBJ_FILE == "your_original_mesh.obj":
# #             print("\n请先编辑此脚本，")
# #             print("将 'your_original_mesh.obj' 修改为您实际的OBJ文件名。")
# #         else:
# #             transform_obj(INPUT_OBJ_FILE, OUTPUT_OBJ_FILE)

# #             # szq_GES_transform_obj.py


# import open3d as o3d
# import numpy as np
# import argparse
# import time

# def align_meshes(source_path, target_path, output_path, voxel_size=1.0):
#     """
#     使用 ICP 算法将源网格对齐到目标网格。

#     :param source_path: 要被变换的网格 (terra.obj)
#     :param target_path: 目标对齐网格 (las.obj)
#     :param output_path: 保存对齐后网格的路径
#     :param voxel_size: 体素大小（米）。用于下采样以加快ICP速度。
#                        根据你网格的尺度，你可能需要调整这个值。
#                        1.0（1米）是一个比较安全的初始值。
#     """
    
#     print(f"Loading target mesh (las): {target_path}")
#     mesh_target = o3d.io.read_triangle_mesh(target_path)
    
#     print(f"Loading source mesh (terra): {source_path}")
#     mesh_source = o3d.io.read_triangle_mesh(source_path)

#     if not mesh_target.has_vertices() or not mesh_source.has_vertices():
#         print("Error: One or both meshes failed to load or are empty.")
#         return

#     print(f"Target vertices: {len(mesh_target.vertices)}")
#     print(f"Source vertices: {len(mesh_source.vertices)}")

#     # 1. 准备点云 (PCD)
#     # 我们使用体素下采样来获取代表性的点云，这能极大加快ICP速度
#     print(f"Downsampling point clouds with voxel_size = {voxel_size}m...")
    
#     # 目标PCD
#     pcd_target = o3d.geometry.PointCloud()
#     pcd_target.points = mesh_target.vertices
#     pcd_target_down = pcd_target.voxel_down_sample(voxel_size)
    
#     # 源PCD
#     pcd_source = o3d.geometry.PointCloud()
#     pcd_source.points = mesh_source.vertices
#     pcd_source_down = pcd_source.voxel_down_sample(voxel_size)

#     print(f"Target points after downsampling: {len(pcd_target_down.points)}")
#     print(f"Source points after downsampling: {len(pcd_source_down.points)}")

#     # 2. 运行 ICP
#     print("Running ICP alignment...")
#     start_time = time.time()
    
#     # 使用一个合理的阈值，例如体素大小的2倍
#     threshold = voxel_size * 2
    
#     # 初始变换（假设它们已经在全局坐标系，只是轴向不同）
#     trans_init = np.identity(4) 
    
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#         pcd_source_down, pcd_target_down, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

#     end_time = time.time()
#     print(f"ICP finished in {end_time - start_time:.2f} seconds.")
#     print("ICP Transformation Matrix:")
#     print(reg_p2p.transformation)

#     # 3. 应用变换
#     print("Applying transformation to the *full* source mesh...")
#     # 我们将计算出的变换应用到 *原始的、高分辨率* 的源网格上
#     mesh_source.transform(reg_p2p.transformation)

#     # 4. 保存结果
#     print(f"Saving aligned mesh to: {output_path}")
#     o3d.io.write_triangle_mesh(output_path, mesh_source)
#     print("Done.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Align one mesh (source) to another (target) using ICP.")
#     parser.add_argument("--source", required=True, help="Path to the source mesh (e.g., terra.obj)")
#     parser.add_argument("--target", required=True, help="Path to the target mesh (e.g., las.obj)")
#     parser.add_argument("--output", required=True, help="Path to save the aligned mesh (e.g., terra_aligned.obj)")
#     parser.add_argument("--voxel_size", type=float, default=1.0, help="Voxel size for downsampling (in meters). Default: 1.0")
    
#     args = parser.parse_args()
    
#     align_meshes(args.source, args.target, args.output, args.voxel_size)


# # python szq_GES_transform_obj.py --source /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/Mergedmesh.obj --target /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/edmseh/cloud_merged.obj --output /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/edmseh/terra_aligned.obj --voxel_size 1.0


import open3d as o3d
import numpy as np
import argparse
import xml.etree.ElementTree as ET
import sys

def get_translation_from_xml(xml_path):
    """
    从 metadata.xml 文件中解析 <SRSOrigin> 标签。
    这个 'translation' 向量是从 "Local" 转到 "Global" 的向量。
    """
    try:
        root = ET.parse(xml_path).getroot()
        srs_origin_element = root.find('SRSOrigin')
        
        if srs_origin_element is None or not srs_origin_element.text:
            print(f"Error: Cannot find <SRSOrigin> tag in {xml_path}")
            return None
            
        translation = np.array(srs_origin_element.text.split(','), dtype=float)
        print(f"Found translation vector in XML: {translation}")
        return translation
        
    except Exception as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return None

def apply_inverse_shift(source_path, xml_path, output_path):
    """
    加载一个 "Global" 网格，减去 'translation' 向量，
    将其转换为 "Local" 网格并保存。
    """
    
    # 1. 获取 translation 向量
    # (Global = Local + translation)
    translation = get_translation_from_xml(xml_path)
    if translation is None:
        sys.exit(1)

    # 2. 加载 "Global" 网格 (terra.obj)
    print(f"Loading global mesh: {source_path}")
    mesh_global = o3d.io.read_triangle_mesh(source_path)
    
    if not mesh_global.has_vertices():
        print(f"Error: Failed to load mesh from {source_path}")
        sys.exit(1)
        
    print(f"Mesh has {len(mesh_global.vertices)} vertices.")

    # 3. 应用逆向变换
    # (Local = Global - translation)
    print("Applying inverse translation (Global -> Local)...")
    
    # 将 translation 转换为 Open3D 期望的格式 (4x4 变换矩阵)
    # 这是一个纯平移矩阵
    transform_matrix = np.identity(4)
    transform_matrix[0:3, 3] = -translation  # 关键：应用负平移
    
    mesh_global.transform(transform_matrix)
    
    # 4. 保存 "Local" 网格
    print(f"Saving local mesh to: {output_path}")
    o3d.io.write_triangle_mesh(output_path, mesh_global)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a 'Global' mesh to a 'Local' mesh by subtracting the <SRSOrigin> vector.")
    parser.add_argument("--source", required=True, help="Path to the 'Global' source mesh (terra.obj)")
    parser.add_argument("--xml", required=True, help="Path to the metadata.xml file containing the <SRSOrigin> tag.")
    parser.add_argument("--output", required=True, help="Path to save the 'Local' mesh (e.g., terra_local.obj)")
    
    args = parser.parse_args()
    
    apply_inverse_shift(args.source, args.xml, args.output)


    '''

python szq_GES_transform_obj.py \
    --source /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/edmseh/Mergedmesh.obj \
    --xml /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/metadata.xml \
    --output /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/edmseh/terra_local.obj


    '''


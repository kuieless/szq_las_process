import open3d as o3d
import numpy as np
import torch
import glob
import os
from tqdm import tqdm
import argparse

def main_cli_debug(args):
    """
    非图形化诊断脚本:
    1. 加载 OBJ 模型并计算其中心点。
    2. 加载所有 NeRF 相机位姿并计算其平均中心点。
    3. 计算并打印出建议的平移向量。
    """
    print(f"Loading mesh from: {args.obj_path}")
    if not os.path.exists(args.obj_path):
        print(f"Error: OBJ file not found at {args.obj_path}")
        return

    mesh = o3d.io.read_triangle_mesh(args.obj_path)
    if not mesh.has_vertices():
        print("Error: The loaded mesh has no vertices.")
        return
        
    mesh_center = mesh.get_center()
    mesh_bbox = mesh.get_axis_aligned_bounding_box()
    
    print("\n--- Mesh Information ---")
    print(f"Mesh Center: {mesh_center}")
    print(f"Mesh Bounding Box (min): {mesh_bbox.min_bound}")
    print(f"Mesh Bounding Box (max): {mesh_bbox.max_bound}")
    
    print("\n--- Camera Information ---")
    metadata_dir = os.path.join(args.nerf_dir, 'val', 'metadata')
    if not os.path.exists(metadata_dir):
        print(f"Error: NeRF metadata directory not found at {metadata_dir}")
        return

    metadata_files = sorted(glob.glob(os.path.join(metadata_dir, '*.pt')))
    if not metadata_files:
        print(f"Error: No metadata .pt files found in {metadata_dir}")
        return

    camera_centers = []
    print(f"Loading {len(metadata_files)} camera poses...")
    for meta_file in tqdm(metadata_files):
        metadata = torch.load(meta_file)
        c2w = metadata['c2w'].numpy()
        
        # 你的主脚本中对相机坐标系进行了Y/Z轴翻转，这里保持一致
        E1 = np.copy(c2w)
        E1[:, 1] *= -1
        E1[:, 2] *= -1
        camera_centers.append(E1[:3, 3])

    avg_camera_center = np.mean(np.array(camera_centers), axis=0)
    cameras_bbox_min = np.min(np.array(camera_centers), axis=0)
    cameras_bbox_max = np.max(np.array(camera_centers), axis=0)

    print(f"\nCamera Cluster Center: {avg_camera_center}")
    print(f"Cameras Bounding Box (min): {cameras_bbox_min}")
    print(f"Cameras Bounding Box (max): {cameras_bbox_max}")
    
    # --- 核心计算 ---
    suggested_translation = avg_camera_center - mesh_center
    
    print("\n" + "="*50)
    print("!!! DIAGNOSIS COMPLETE !!!")
    print("="*50)
    print("The mesh center is very far from the camera cluster center.")
    print("To align them, you need to translate the mesh vertices.")
    print("\nSUGGESTED TRANSLATION VECTOR (avg_camera_center - mesh_center):")
    print(f"===>  np.array([{suggested_translation[0]:.6f}, {suggested_translation[1]:.6f}, {suggested_translation[2]:.6f}])  <===")
    print("\nInstructions: Copy the vector above and apply it in your main script's 'load_mesh' function.")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLI tool to diagnose mesh and camera alignment.")
    parser.add_argument('--obj_path', type=str, required=True, help='Path to the .obj file.')
    parser.add_argument('--nerf_dir', type=str, required=True, help='Path to the root NeRF dataset directory.')
    args = parser.parse_args()
    main_cli_debug(args)

'''
python debug_cli.py \
--obj_path /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/merged_model.obj \
--nerf_dir /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7

'''
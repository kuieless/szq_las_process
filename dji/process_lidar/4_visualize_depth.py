import os
import numpy as np
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from glob import glob
from pyntcloud import PyntCloud
import argparse

def create_point_cloud(args):
    """
    Loads depth maps and RGB images from a dataset folder,
    projects them into a 3D point cloud, and saves it as a .ply file.
    This script is a standalone replacement and does not require the gp_nerf framework.
    """
    # Define paths based on the dataset directory
    # The dataset path should point to the directory containing 'rgbs', 'metadata', etc.
    rgb_dir = os.path.join(args.dataset_path, 'rgbs')
    metadata_dir = os.path.join(args.dataset_path, 'metadata')
    
    if args.depth_type == 'mesh':
        depth_dir = os.path.join(args.dataset_path, 'depth_mesh')
    elif args.depth_type == 'lidar':
        # Assuming LiDAR depth is in a folder named 'depth_dji' as per the original script
        depth_dir = os.path.join(args.dataset_path, 'depth_dji')
    else:
        raise ValueError(f"Unsupported depth_type: {args.depth_type}")

    print(f"Reading RGBs from: {rgb_dir}")
    print(f"Reading depths from: {depth_dir}")
    print(f"Reading metadata from: {metadata_dir}")

    # Check if directories exist
    for path in [rgb_dir, metadata_dir, depth_dir]:
        if not os.path.isdir(path):
            print(f"Error: Directory not found at '{path}'")
            return

    rgb_files = sorted(glob(os.path.join(rgb_dir, '*.jpg')))
    if not rgb_files:
        print(f"Error: No .jpg files found in {rgb_dir}")
        return

    world_points = []
    for rgb_path in tqdm(rgb_files, desc=f"Processing {args.depth_type} depth maps"):
        base_name = os.path.splitext(os.path.basename(rgb_path))[0]
        
        # Construct paths to corresponding files
        depth_path = os.path.join(depth_dir, f'{base_name}.npy')
        metadata_path = os.path.join(metadata_dir, f'{base_name}.pt')

        if not (os.path.exists(depth_path) and os.path.exists(metadata_path)):
            print(f"Warning: Skipping {base_name}, missing depth or metadata file.")
            continue

        # 1. Load data
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB) # Convert to RGB
        depth_map = np.load(depth_path).squeeze()
        metadata = torch.load(metadata_path)
        intrinsics = metadata['intrinsics'].numpy() # [fx, fy, cx, cy]
        c2w = metadata['c2w'].numpy() # Camera-to-world matrix

        # 2. Core projection logic (from your original script)
        H, W = depth_map.shape
        valid_depth_mask = ~np.isinf(depth_map) & (depth_map > 0)

        # Build intrinsics matrix K
        K = np.array([[intrinsics[0], 0, intrinsics[2]],
                      [0, intrinsics[1], intrinsics[3]],
                      [0, 0, 1]])
        K_inv = np.linalg.inv(K)

        # Create pixel coordinates
        y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        pixel_coordinates = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)

        # Project 2D pixels to 3D points in camera coordinates
        # P_cam = D * K_inv @ p
        pt_cam = depth_map[..., np.newaxis] * (K_inv @ pixel_coordinates[..., np.newaxis]).squeeze()
        
        # Add homogeneous coordinate
        pt_cam_hom = np.concatenate([pt_cam, np.ones((H, W, 1))], axis=-1)
        
        # Flatten and filter for valid points
        pt_cam_hom_flat = pt_cam_hom[valid_depth_mask]
        rgb_flat = rgb_image[valid_depth_mask]

        # Transform points to world coordinates using c2w
        # This coordinate system flip is specific to the original NeRF pipeline
        # E = np.copy(c2w)
        # E[:, 1] *= -1
        # E[:, 2] *= -1
        
        # pt_world_hom = (E @ pt_cam_hom_flat.T).T
        # pt_world = pt_world_hom[:, :3] / pt_world_hom[:, 3:4]

        # Transform points to world coordinates using c2w
        # This coordinate system flip is specific to the original NeRF pipeline
        E_3x4 = np.copy(c2w)
        E_3x4[:, 1] *= -1
        E_3x4[:, 2] *= -1
        
        # Augment the 3x4 matrix to a 4x4 homogeneous transformation matrix
        E_4x4 = np.eye(4)
        E_4x4[:3, :] = E_3x4
        
        # Transform points using the full 4x4 matrix
        pt_world_hom = (E_4x4 @ pt_cam_hom_flat.T).T
        
        # Now, perform the perspective division (de-homogenization). This will now work.
        pt_world = pt_world_hom[:, :3] / pt_world_hom[:, 3:4]

        # Combine 3D points and RGB colors
        pc_rgb = np.hstack((pt_world, rgb_flat))
        world_points.append(pc_rgb)

    if not world_points:
        print("No points were processed. Exiting.")
        return

    # Combine all points into a single array
    combined_array = np.concatenate(world_points, axis=0)
    print(f"Total points generated: {combined_array.shape[0]}")

    # 3. Save to .ply file
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.join(args.output_dir, f"point_cloud_{args.depth_type}.ply")
    print(f"Saving point cloud to {output_filename} ...")

    # PyntCloud expects colors in range [0, 1]
    cloud_data = pd.DataFrame(
        data=np.hstack((combined_array[:, :3], combined_array[:, 3:6] / 255.0)),
        columns=["x", "y", "z", "red", "green", "blue"]
    )
    cloud = PyntCloud(cloud_data)
    cloud.to_file(output_filename)
    
    print("Done.")

def _get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help="Path to the dataset directory (e.g., '.../output7/val')")
    parser.add_argument('--depth_type', type=str, required=True, choices=['mesh', 'lidar'],
                        help="Which type of depth maps to process.")
    parser.add_argument('--output_dir', type=str, default='point_clouds',
                        help="Directory to save the output .ply files.")
    return parser.parse_args()

if __name__ == '__main__':
    args = _get_opts()
    create_point_cloud(args)

'''
python 4_visualize_depth.py --dataset_path /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val --depth_type mesh


python 4_visualize_depth.py --dataset_path /data/yuqi/Datasets/DJI/DJI_20230726_xiayuan2 --depth_type lidar

'''
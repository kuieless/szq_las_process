## 这个代码用来获取residence数据集中的subdataset
## 我们利用gp_nerf/datasets/memory_dataset_sam_project_one_point.py选择了某一建筑物上的单点，将其投影到其他视角中，获取一个子集
## 这个数据保存在zyq/project5_one_building
## 需要对这个子集进行提取

import glob
import os
import numpy as np
import re
import shutil
from pathlib import Path
import torch
import shutil
from dji.visual_poses import load_poses
def main() -> None:

    image_folder = "zyq/project_subset_959/sample/"
    dataset_path = "/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan"
    input_folder_train = os.path.join(dataset_path, "train")
    input_folder_val = os.path.join(dataset_path, "val")
    
    output_folder = "Output_subset/"
    if not os.path.exists(output_folder):
        Path(output_folder).mkdir()
    image_paths = glob.glob(image_folder + "*.jpg")
    indices = []
    

    for image_path in image_paths:
        filename = image_path.split("/")[-1]  # 获取文件名部分
        index_str = filename.split("_")[0]  # 分割后获取索引部分
        index_int = int(index_str)  # 转换为整数
        indices.append(index_int)
    sorted_indices = sorted(indices)

    train_paths = sorted(os.listdir(os.path.join(dataset_path, "train", "metadata")))
    val_paths = sorted(os.listdir(os.path.join(dataset_path, "val", "metadata")))


    # 提取文件名中的数字部分并转换为整数
    train_indices = []
    for file_name in train_paths:
        if file_name.endswith(".pt"):
            index_str = file_name.split(".")[0]  # 去除文件扩展名
            index_int = int(index_str)
            train_indices.append(index_int)
    
    # 提取文件名中的数字部分并转换为整数
    val_indices = []
    for file_name in val_paths:
        if file_name.endswith(".pt"):
            index_str = file_name.split(".")[0]  # 去除文件扩展名
            index_int = int(index_str)
            val_indices.append(index_int)
    
    for index in sorted_indices:
        split = 'train'

        for source_folder in ['rgbs', 'metadata']:
            if source_folder == 'rgbs':
                extend_name = '.jpg'
            elif source_folder == 'metadata':
                extend_name = '.pt'

            source_file = os.path.join(dataset_path, split, source_folder, f"{index:06d}{extend_name}")

            # 判断文件是否存在，存在则复制到输出目录
            if os.path.exists(source_file):
                output_ext_folder = os.path.join(output_folder, split, source_folder)

                # 确保输出文件夹存在
                Path(output_ext_folder).mkdir(parents=True, exist_ok=True)

                # 复制文件到输出目录
                shutil.copy(source_file, os.path.join(output_ext_folder, f"{index:06d}{extend_name}"))

    print("Files copied and saved.")


    #####以上写了N个pose#################

    
    # 这里只读了train
    selected_poses = load_poses(dataset_path=output_folder)
    
    #将pose转变为原尺度
    coordinates_old = torch.load(os.path.join(dataset_path,'coordinates.pt'))
    origin_old = coordinates_old['origin_drb'].numpy()
    pose_scale_factor = coordinates_old['pose_scale_factor']
    selected_poses = np.array(selected_poses)
    selected_poses[:,:, 3] = (selected_poses[:,:, 3] * pose_scale_factor) + origin_old


    # 过滤掉某些较远的pose
    camera_positions = selected_poses[:,1:3,3]
    min_position = np.min(camera_positions,axis=0)
    max_position = np.max(camera_positions,axis=0)
    bianchang = (max_position- min_position) /8
    min_bound = min_position * 1
    max_bound = max_position * 1
    center_camera =np.where(
        (camera_positions[:, 0] >= min_bound[0]) & (camera_positions[:, 0] <= max_bound[0]) &
        (camera_positions[:, 1] >= min_bound[1]) & (camera_positions[:, 1] <= max_bound[1])
    )
    center_camera=list(center_camera)
    center_camera = sorted(center_camera)


    selected_poses = [selected_poses[int(index)] for index in center_camera[0]]
    selected_poses = np.array(selected_poses)


    ## 过滤后的pose重新归一化 
    min_position = np.min(selected_poses[:,:, 3], axis=0)
    max_position = np.max(selected_poses[:,:, 3], axis=0)
    print('Coord range: {} {}'.format(min_position, max_position))
    origin = (max_position + min_position) * 0.5
    dist = (torch.tensor(selected_poses[:,:, 3]) - torch.tensor(origin)).norm(dim=-1)
    diagonal = dist.max()
    scale = diagonal.numpy()
    selected_poses[:,:, 3] = (selected_poses[:,:, 3] - origin) / scale

    coordinates = {
        'origin_drb': torch.Tensor(origin),
        'pose_scale_factor': scale
    }
    if not os.path.exists(output_folder):
        Path(output_folder).mkdir()
    torch.save(coordinates, os.path.join(output_folder, 'coordinates.pt'))

    ## 清空文件夹，重新写
    if os.path.exists(os.path.join(output_folder, 'train')):
        shutil.rmtree(os.path.join(output_folder, 'train'))  # 递归删除文件夹及其内容
    if os.path.exists(os.path.join(output_folder, 'val')):
        shutil.rmtree(os.path.join(output_folder, 'val'))  # 递归删除文件夹及其内容

    print("Folder cleared.")

    sorted_indices = [sorted_indices[int(index)] for index in center_camera[0]]
    for i, index in enumerate(sorted_indices):
        if index in train_indices:
            split = 'train'
        else:
            split = 'val'

        for source_folder in ['rgbs', 'metadata', 'depth_mesh']:
            if source_folder == 'rgbs':
                extend_name = '.jpg'
            
                source_file = os.path.join(dataset_path, split, source_folder, f"{index:06d}{extend_name}")

                # 判断文件是否存在，存在则复制到输出目录
                if os.path.exists(source_file):
                    output_ext_folder = os.path.join(output_folder, split, source_folder)
                    # 确保输出文件夹存在
                    Path(output_ext_folder).mkdir(parents=True, exist_ok=True)
                    # 复制文件到输出目录
                    shutil.copy(source_file, os.path.join(output_ext_folder, f"{index:06d}{extend_name}"))

            elif source_folder == 'metadata':
                extend_name = '.pt'

                source_file = os.path.join(dataset_path, split, source_folder, f"{index:06d}{extend_name}")

                metadata_old = torch.load(source_file)
                metadata_old['c2w'] = torch.FloatTensor(selected_poses[i])
                
                # 判断文件是否存在，存在则复制到输出目录
                if os.path.exists(source_file):
                    output_ext_folder = os.path.join(output_folder, split, source_folder)
                    # 确保输出文件夹存在
                    Path(output_ext_folder).mkdir(parents=True, exist_ok=True)
                    torch.save(metadata_old, os.path.join(output_ext_folder, f"{index:06d}{extend_name}"))
            elif source_folder == 'depth_mesh':
                extend_name = '.npy'
            
                source_file = os.path.join(dataset_path, split, source_folder, f"{index:06d}{extend_name}")

                # 判断文件是否存在，存在则复制到输出目录
                if os.path.exists(source_file):
                    output_ext_folder = os.path.join(output_folder, split, source_folder)
                    # 确保输出文件夹存在
                    Path(output_ext_folder).mkdir(parents=True, exist_ok=True)
                    # 复制文件到输出目录
                    shutil.copy(source_file, os.path.join(output_ext_folder, f"{index:06d}{extend_name}"))

    print("Files copied and saved.")

    a = 1 
    


if __name__ == '__main__':
    main()
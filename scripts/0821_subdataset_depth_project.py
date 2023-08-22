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
def main() -> None:

    image_folder = "zyq/project5_one_building/sample/"
    dataset_path = "/data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels"
    input_folder_train = os.path.join(dataset_path, "train")
    input_folder_val = os.path.join(dataset_path, "val")
    
    output_folder = "Output_subset/"




    image_paths = glob.glob(image_folder + "*.jpg")
    indices = []
    

    for image_path in image_paths:
        filename = image_path.split("/")[-1]  # 获取文件名部分
        index_str = filename.split("_")[1]  # 分割后获取索引部分
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
        if index in train_indices:
            split = 'train'
        else:
            split = 'val'

        for source_folder in ['rgbs', 'metadata']:
            if source_folder == 'rgbs':
                extend_name = '.JPG'
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

    a = 1 
    


if __name__ == '__main__':
    main()
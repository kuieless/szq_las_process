#!/bin/bash

# 源文件夹A和目标文件夹B的路径
folderA="/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.5/pred_depth_save"
folderB="/data/yuqi/Datasets/DJI/Yingrenshi_20230926_subset/train/rgbs"

# 遍历文件夹A内的文件
for fileA in "$folderA"/*; do
    # 获取文件名（不带后缀）
    filenameA=$(basename "$fileA" | cut -d'_' -f1)
    echo $filenameA
    
    # 检查文件名是否存在于文件夹B
    if [ -e "$folderB/$filenameA.jpg" ]; then
        echo "File $filenameA found in folderB, keeping."
    else
        echo "File $filenameA not found in folderB, deleting."
        # 删除文件
        rm -f "$fileA"
    fi
done


# # 源文件夹A和目标文件夹B的路径
# folderA="/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.5/pred_rgb"
# folderB="/data/yuqi/Datasets/DJI/Yingrenshi_20230926_subset/train/rgbs"

# # 遍历文件夹A内的文件
# for fileA in "$folderA"/*; do
#     # 获取文件名（不带后缀）
#     filenameA=$(basename "$fileA" | cut -d'_' -f1)
#     echo $filenameA
    
#     # 检查文件名是否存在于文件夹B
#     if [ -e "$folderB/$filenameA" ]; then
#         echo "File $filenameA found in folderB, keeping."
#     else
#         echo "File $filenameA not found in folderB, deleting."
#         # 删除文件
#         rm -f "$fileA"
#     fi
# done

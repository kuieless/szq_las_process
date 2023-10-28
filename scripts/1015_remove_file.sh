#!/bin/bash

# # 源文件夹A和目标文件夹B的路径
# folderA="/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.5/pred_depth_save"
# folderB="/data/yuqi/Datasets/DJI/Yingrenshi_20230926_subset/train/rgbs"

# # 遍历文件夹A内的文件
# for fileA in "$folderA"/*; do
#     # 获取文件名（不带后缀）
#     filenameA=$(basename "$fileA" | cut -d'_' -f1)
#     echo $filenameA
    
#     # 检查文件名是否存在于文件夹B
#     if [ -e "$folderB/$filenameA.jpg" ]; then
#         echo "File $filenameA found in folderB, keeping."
#     else
#         echo "File $filenameA not found in folderB, deleting."
#         # 删除文件
#         rm -f "$fileA"
#     fi
# done


# 源文件夹A和目标文件夹B的路径
folderA="/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1028_lh_block1_far0.3/1_labels_m2f_only_sam/labels_merge_vis_alpha"
folderB="/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1028_lh_block1_far0.3/2_project_to_ori_gt_only_sam/alpha"

# 遍历文件夹A内的文件
for fileA in "$folderA"/*; do
    # 获取文件名（不带后缀）
    filenameA=$(basename "$fileA" | cut -d'.' -f1)
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
# folderA="/data/yuqi/code/Mask2Former_a/output_useful/1016_compare_ds2/labels_m2f_ds4"
# folderB="/data/yuqi/Datasets/DJI/Yingrenshi_20230926_subset/train/rgbs"

# # 遍历文件夹A内的文件
# for fileA in "$folderA"/*; do
#     # 获取文件名（不带后缀）
#     filenameA=$(basename "$fileA" | cut -d'_' -f1)
#     echo $filenameA
    
#     # # 检查文件名是否存在于文件夹B
#     # if [ -e "$folderB/$filenameA" ]; then
#     #     echo "File $filenameA found in folderB, keeping."
#     # else
#     #     echo "File $filenameA not found in folderB, deleting."
#     #     # 删除文件
#     #     rm -f "$fileA"
#     # fi
# done

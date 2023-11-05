#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata     labels_m2f
# .1npy        .1jpg   .1pt                             .1png
source_dir="/data/yuqi/Datasets/DJI/Longhua_block1_20231009/train/depth_mesh"
destination_dir="/data/yuqi/Datasets/DJI/Longhua_block1_20231009/val/depth_mesh"
filenames=("000063.npy" "000217.npy" "000243.npy" "000483.npy" "000529.npy" "000711.npy" "000758.npy" "000872.npy")  # 在这里列出你想要移动的文件名


# 进入源文件夹
cd "$source_dir" || exit

# 移动指定文件名的文件到目标文件夹
for filename in "${filenames[@]}"; do
    if [ -e "$filename" ]; then
        sudo mv "$filename" "$destination_dir"
        echo "已移动文件: $destination_dir/$filename"
    else
        echo "未找到文件: $filename"
    fi
done

echo "移动完成"

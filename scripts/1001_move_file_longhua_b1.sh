#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata     label_m2f
# .1npy        .1jpg   .1pt                             .1pt
source_dir="/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds/train/depth_mesh"
destination_dir="/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds/val/depth_mesh"
filenames=("000148.npy" "000168.npy" "000172.npy" "000183.npy" "000328.npy" "000483.npy" "000529.npy" "000548.npy" "000612.npy" "000647.npy" "000711.npy" "000758.npy" "000759.npy" "000872.npy" "000902.npy")  # 在这里列出你想要移动的文件名


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

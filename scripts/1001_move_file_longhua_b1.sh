#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata     label_m2f
# .1npy        .1jpg   .1pt                             .1pt
source_dir="/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds/train/metadata"
destination_dir="/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds/val/metadata"
filenames=("000148.pt" "000168.pt" "000172.pt" "000183.pt" "000328.pt" "000483.pt" "000529.pt" "000548.pt" "000612.pt" "000647.pt" "000711.pt" "000758.pt" "000759.pt" "000872.pt" "000902.pt")  # 在这里列出你想要移动的文件名


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

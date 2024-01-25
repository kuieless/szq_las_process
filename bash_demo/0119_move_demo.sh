#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata     label_m2f
# .1npy        .1jpg   .1pt                             .1pt

source_dir="/data/yuqi/Datasets/DJI/xiayuan_20240118_demo/val/rgbs"
destination_dir="/data/yuqi/Datasets/DJI/xiayuan_20240118_demo/train/rgbs"
filenames=("000000.jpg" "000062.jpg" "000186.jpg" "000558.jpg" "000620.jpg" "000744.jpg" "000930.jpg" "000992.jpg" "001178.jpg" "001240.jpg")


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

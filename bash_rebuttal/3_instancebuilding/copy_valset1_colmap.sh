#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata
# .1npy        .1jpg   .1pt     
source_dir="/data/yuqi/Datasets/InstanceBuilding/3D/scene1/colmap_jx/train/rgbs"
destination_dir="/data/yuqi/Datasets/InstanceBuilding/3D/scene1/colmap_jx/val/rgbs"
filenames=("000005.jpg" "000010.jpg" "000015.jpg" "000020.jpg" "000025.jpg" "000030.jpg" "000035.jpg" "000040.jpg" "000045.jpg" "000050.jpg"  "000055.jpg" "000060.jpg" "000065.jpg" "000070.jpg" "000075.jpg")

# 进入源文件夹
cd "$source_dir" || exit

# 复制并重命名指定文件名的文件到目标文件夹
for filename in "${filenames[@]}"; do
    if [ -e "$filename" ]; then
        new_filename=$(printf "%06d.jpg" "$((10#${filename%%.jpg} + 100))")
        cp "$filename" "$destination_dir/$new_filename"
        echo "已复制文件: $destination_dir/$new_filename"
    else
        echo "未找到文件: $filename"
    fi
done

echo "复制完成"


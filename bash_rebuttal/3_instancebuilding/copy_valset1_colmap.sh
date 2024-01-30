#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata
# .1npy        .1jpg   .1pt     
source_dir="/data/yuqi/Datasets/InstanceBuilding/3D/scene1/colmap_jx_new/train/labels_gt"
destination_dir="/data/yuqi/Datasets/InstanceBuilding/3D/scene1/colmap_jx_new/val/labels_gt"
filenames=("000005.png" "000010.png" "000015.png" "000020.png" "000025.png" "000030.png" "000035.png" "000040.png" "000045.png" "000050.png"  "000055.png" "000060.png" "000065.png" "000070.png" "000075.png")

# 进入源文件夹
cd "$source_dir" || exit

# 复制并重命名指定文件名的文件到目标文件夹
for filename in "${filenames[@]}"; do
    if [ -e "$filename" ]; then
        new_filename=$(printf "%06d.png" "$((10#${filename%%.png} + 100))")
        cp "$filename" "$destination_dir/$new_filename"
        echo "已复制文件: $destination_dir/$new_filename"
    else
        echo "未找到文件: $filename"
    fi
done

echo "复制完成"


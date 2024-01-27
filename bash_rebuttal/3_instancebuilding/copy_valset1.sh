#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata
# .1npy        .1jpg   .1pt     
source_dir="/data/yuqi/Datasets/InstanceBuilding/3D/scene1/output/train/metadata"
destination_dir="/data/yuqi/Datasets/InstanceBuilding/3D/scene1/output/val/metadata"
filenames=("000005.pt" "000010.pt" "000015.pt" "000020.pt" "000025.pt" "000030.pt" "000035.pt" "000040.pt" "000045.pt" "000050.pt"  "000055.pt" "000060.pt" "000065.pt" "000070.pt" "000075.pt")

# 进入源文件夹
cd "$source_dir" || exit

# 复制并重命名指定文件名的文件到目标文件夹
for filename in "${filenames[@]}"; do
    if [ -e "$filename" ]; then
        new_filename=$(printf "%06d.pt" "$((10#${filename%%.pt} + 100))")
        cp "$filename" "$destination_dir/$new_filename"
        echo "已复制文件: $destination_dir/$new_filename"
    else
        echo "未找到文件: $filename"
    fi
done

echo "复制完成"


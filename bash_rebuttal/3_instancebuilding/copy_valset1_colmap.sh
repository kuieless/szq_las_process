#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata
# .1npy        .1jpg   .1pt     
source_dir="/data/yuqi/Datasets/InstanceBuilding/3D/scene1/colmap_jx_new/train/instances_mask_0.001_depth20_crossview_process"
destination_dir="/data/yuqi/Datasets/InstanceBuilding/3D/scene1/colmap_jx_new/val/instances_mask_0.001_depth20_crossview_process"
filenames=("000005.npy" "000010.npy" "000015.npy" "000020.npy" "000025.npy" "000030.npy" "000035.npy" "000040.npy" "000045.npy" "000050.npy"  "000055.npy" "000060.npy" "000065.npy" "000070.npy" "000075.npy")

# 进入源文件夹
cd "$source_dir" || exit

# 复制并重命名指定文件名的文件到目标文件夹
for filename in "${filenames[@]}"; do
    if [ -e "$filename" ]; then
        new_filename=$(printf "%06d.npy" "$((10#${filename%%.npy} + 100))")
        cp "$filename" "$destination_dir/$new_filename"
        echo "已复制文件: $destination_dir/$new_filename"
    else
        echo "未找到文件: $filename"
    fi
done

echo "复制完成"


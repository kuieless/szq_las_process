#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata
# .1npy        .1jpg   .1pt     
source_dir="/data/yuqi/Datasets/InstanceBuilding/3D/scene2/output/train/rgbs"
destination_dir="/data/yuqi/Datasets/InstanceBuilding/3D/scene2/output/val/rgbs"
filenames=("000003.jpg" "000014.jpg" "000017.jpg" "000031.jpg" "000037.jpg" "000050.jpg" "000053.jpg" "000058.jpg" "000009.jpg" "000012.jpg"  "000020.jpg" "000032.jpg" "000044.jpg" "000046.jpg" "000052.jpg")

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


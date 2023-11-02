#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata     labels_m2f
# .1npy        .1jpg   .1pt                             .1png
source_dir="/data/yuqi/Datasets/DJI/Campus/train/labels_m2f"
destination_dir="/data/yuqi/Datasets/DJI/Campus/val/labels_m2f"
filenames=("000187.png" "000197.png" "000242.png" "000277.png" "000376.png" "000380.png" "000445.png" "000460.png" "000513.png" "000554.png" "000750.png" "000802.png")  # 在这里列出你想要移动的文件名


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

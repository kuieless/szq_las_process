#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata     labels_m2f
# .1npy        .1jpg   .1pt                             .1png
source_dir="/data/yuqi/Datasets/DJI/Longhua_block2_20231017/train/rgbs"
destination_dir="/data/yuqi/Datasets/DJI/Longhua_block2_20231017/val/rgbs"
filenames=("000002.jpg" "000138.jpg" "000210.jpg" "000237.jpg" "000288.jpg" "000323.jpg" "000475.jpg" "000479.jpg" "000532.jpg" "000550.jpg" )  # 在这里列出你想要移动的文件名


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

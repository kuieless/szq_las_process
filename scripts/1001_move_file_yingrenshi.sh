#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata     label_m2f
# .1npy        .1jpg   .1pt                             .png
source_dir="/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/instances_gt"
destination_dir="/data/yuqi/Datasets/DJI/Yingrenshi_20230926/val/instances_gt"
filenames=("000092.png" "000114.png" "000145.png" "000172.png" "000204.png" "000231.png" "000266.png" "000295.png" "000327.png" "000353.png" "000385.png" "000409.png" "000777.png" "000802.png" "000832.png")  # 在这里列出你想要移动的文件名


# 进入源文件夹
cd "$source_dir" || exit

# 移动指定文件名的文件到目标文件夹
for filename in "${filenames[@]}"; do
    if [ -e "$filename" ]; then
        mv "$filename" "$destination_dir"
        echo "已移动文件: $destination_dir/$filename"
    else
        echo "未找到文件: $filename"
    fi
done

echo "移动完成"

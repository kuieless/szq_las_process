#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata     labels_m2f
# .1npy        .1jpg   .1pt                             .1png
source_dir="/data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds/train/labels_gt"
destination_dir="/data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds/val/labels_gt"
filenames=("000002.png" "000138.png" "000210.png" "000237.png" "000288.png" "000323.png" "000475.png" "000479.png" "000532.png" "000550.png" )  # 在这里列出你想要移动的文件名


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

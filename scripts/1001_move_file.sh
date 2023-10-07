#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata
# .1npy        .1jpg   .1pt     
source_dir="/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/train/image_metadata"
destination_dir="/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/val/image_metadata"
filenames=("000082.pt" "000098.pt" "000190.pt" "000202.pt" "000350.pt" "000438.pt" "000496.pt" "000501.pt" "000561.pt" "000815.pt" "000879.pt" "000961.pt" "001010.pt" "001037.pt" "001352.pt" "001384.pt" "001414.pt" "001615.pt" "001733.pt" "001834.pt" "001926.pt")  # 在这里列出你想要移动的文件名

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

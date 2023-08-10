#!/bin/bash
# .1npy    .1pt  .1jpg
source_dir="/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/train/rgbs"
destination_dir="/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/val/rgbs"
filenames=("000082.jpg" "000098.jpg" "000190.jpg" "000202.jpg" "000350.jpg" "000438.jpg" "000496.jpg" "000501.jpg" "000561.jpg" "000815.jpg" "000879.jpg" "000961.jpg" "001010.jpg" "001037.jpg" "001352.jpg" "001384.jpg" "001414.jpg" "001615.jpg" "001733.jpg" "001834.jpg" "001926.jpg")  # 在这里列出你想要移动的文件名

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

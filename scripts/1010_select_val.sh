#!/bin/bash
#  depth_mesh  rgbs     image_metadata    metadata     label_m2f
# .1npy        .1jpg   .1pt                             .1png
source_dir="/data/jxchen/dataset/dji/longhua/render_dji_ply_meshMask_m2fLabel/alphaCover_thres10"
destination_dir="/data/jxchen/dataset/dji/longhua/render_dji_ply_meshMask_m2fLabel/alphaCover_thres10_zyq_select"
filenames=("000148.jpg" "000168.jpg" "000183.jpg" "000328.jpg" "000538.jpg" "000612.jpg" "000676.jpg" "000711.jpg" "000734.jpg" "000746.jpg" "000758.jpg" "000772.jpg" "000883.jpg" "000902.jpg" "000158.jpg" "000172.jpg" "000388.jpg" "000483.jpg" "000529.jpg" "000548.jpg" "000612.jpg" "000647.jpg" "000740.jpg" "000759.jpg" "000872.jpg")  # 在这里列出你想要移动的文件名


# 进入源文件夹
cd "$source_dir" || exit

# 移动指定文件名的文件到目标文件夹
for filename in "${filenames[@]}"; do
    if [ -e "$filename" ]; then
        cp "$filename" "$destination_dir"
        echo "已移动文件: $destination_dir/$filename"
    else
        echo "未找到文件: $filename"
    fi
done

echo "移动完成"

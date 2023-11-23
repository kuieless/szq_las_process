#!/bin/bash

# 设置A文件夹路径
input_folder="A_Supplementary/render_instance/yingrenshi"


# 创建MP4文件夹（如果不存在）
mp4_folder="A_Supplementary/render_instance/MP4/yingrenshi"
mkdir -p "$mp4_folder"


# 遍历A文件夹下的所有子文件夹B\C\D
for dir_path in "$input_folder"/*/; do
    dir_name=$(basename "$dir_path")
    
    # 检查是否包含子文件夹B\0\eval_100000\panoptic
    if [ -d "$dir_path/0/eval_100000/panoptic" ]; then
        panoptic_folder="$dir_path/0/eval_100000/panoptic"
        
        # 获取输出视频文件的路径
        output_video_path="$mp4_folder/$dir_name.mp4"

        # 使用ffmpeg将图像转换为视频
        ffmpeg -framerate 24 -pattern_type glob -i "$panoptic_folder/*.jpg" -c:v libx264 -b:v 20M -pix_fmt yuv420p -vf "crop=1368:912:2742:2"  "$output_video_path"
        echo "Video '$output_video_path' created."
    fi
done


export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=6

# crop=-----1368:912:2742:2
# "crop=342A:228:690:2"


# # # ## ours
/home/yuqi/anaconda3/envs/gpnerf/bin/python gp_nerf/eval.py --val_type=train_instance --supp_name=render_supp_tra  --val_scale_factor=4 --cached_centroids_path='zyq/1114_cluster_campus/mean_cross_view_all/mean_depth_200000_1000/test_centroids.npy'    --exp_name A_Supplementary_trajectory/campus --enable_semantic True --config_file configs/campus_new.yaml --batch_size 16384 --train_iterations 100000 --val_interval 50000 --ckpt_interval 50000 --dataset_type memory_depth_dji_instance_crossview --freeze_geo=True --ckpt_path=logs_campus/1104_campus_density_depth_hash22_fusion1018_car2_semantic/0/models/200000.pt --use_subset=True --lr=0.01 --enable_instance=True --freeze_semantic=True --instance_name=instances_mask_0.001_depth --instance_loss_mode=slow_fast --num_instance_classes=25 --crossview_all=True --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_campus/1113_campus_density_depth_hash22_instance_origin_sam_0.001_depth_crossview_all/0/models/100000.pt

#!/bin/bash
# 设置A文件夹路径
input_folder=A_Supplementary_trajectory/campus/0
output_video=A_Supplementary_trajectory/0_campus.mp4


# 1. 生成instance视频
# 遍历A文件夹下的所有子文件夹B\C\D
for dir_path in "$input_folder"/*/; do
    dir_name=$(basename "$dir_path")
    echo $dir_name
    # 检查是否包含子文件夹B\0\eval_100000\panoptic
    if [ -d "$dir_path/panoptic" ]; then
        panoptic_folder="$dir_path/panoptic"
        output_video_path="$input_folder/instance.mp4"
        # 使用ffmpeg将图像转换为视频
        ffmpeg -framerate 24 -pattern_type glob -i "$panoptic_folder/*.jpg" -c:v libx264 -b:v 20M -pix_fmt yuv420p -vf "crop=1368:912:2742:2"  "$output_video_path" -y
        echo "Video '$output_video_path' created."
    fi
done


# 2. 生成 rgb 视频
# 遍历A文件夹下的所有子文件夹B\C\D
for dir_path in "$input_folder"/*/; do
    dir_name=$(basename "$dir_path")
    
    # 检查是否包含子文件夹B\0\eval_100000\panoptic
    if [ -d "$dir_path/val_rgbs/pred_rgb" ]; then
        panoptic_folder="$dir_path/val_rgbs/pred_rgb"
        output_video_path="$input_folder/pred_rgb.mp4"
        # 使用ffmpeg将图像转换为视频
        ffmpeg -framerate 24 -pattern_type glob -i "$panoptic_folder/*.jpg" -c:v libx264 -b:v 20M -pix_fmt yuv420p -vf "crop=1368:912:2742:2"  "$output_video_path" -y
        echo "Video '$output_video_path' created."
    fi
done


# 2. 生成 rgb 视频
# 遍历A文件夹下的所有子文件夹B\C\D
for dir_path in "$input_folder"/*/; do
    dir_name=$(basename "$dir_path")
    
    # 检查是否包含子文件夹B\0\eval_100000\panoptic
    if [ -d "$dir_path/val_rgbs/pred_depth" ]; then
        panoptic_folder="$dir_path/val_rgbs/pred_depth"
        output_video_path="$input_folder/pred_depth.mp4"
        # 使用ffmpeg将图像转换为视频
        ffmpeg -framerate 24 -pattern_type glob -i "$panoptic_folder/*.jpg" -c:v libx264 -b:v 20M -pix_fmt yuv420p -vf "crop=1368:912:2742:2"  "$output_video_path" -y
        echo "Video '$output_video_path' created."
    fi
done

pred_label_alpha

# 2. 生成 rgb 视频
# 遍历A文件夹下的所有子文件夹B\C\D
for dir_path in "$input_folder"/*/; do
    dir_name=$(basename "$dir_path")
    
    # 检查是否包含子文件夹B\0\eval_100000\panoptic
    if [ -d "$dir_path/val_rgbs/pred_label_alpha" ]; then
        panoptic_folder="$dir_path/val_rgbs/pred_label_alpha"
        output_video_path="$input_folder/pred_label_alpha.mp4"
        # 使用ffmpeg将图像转换为视频
        ffmpeg -framerate 24 -pattern_type glob -i "$panoptic_folder/*.jpg" -c:v libx264 -b:v 20M -pix_fmt yuv420p -vf "crop=1368:912:2742:2"  "$output_video_path" -y
        echo "Video '$output_video_path' created."
    fi
done


##拼接视频


# 输入视频
v1=$input_folder/pred_rgb.mp4
v2=$input_folder/pred_depth.mp4
v3=$input_folder/pred_label_alpha.mp4
v4=$input_folder/instance.mp4

# 设置间隔高度
interval_height_top=20
interval_height_bottom=150

# 设置间隔颜色（白色）
interval_color=white

# 设置左上角和右上角、左下角和右下角之间的间隔
interval_width=10

# 拼接filter
filter_complex="
[0:v]setpts=PTS-STARTPTS,pad=iw+$interval_width:ih+$interval_width:$interval_width/2:$interval_width/2:color=$interval_color [v1];
[1:v]setpts=PTS-STARTPTS,pad=iw+$interval_width:ih+$interval_width:$interval_width/2:$interval_width/2:color=$interval_color [v2]; 
[2:v]setpts=PTS-STARTPTS,pad=iw+$interval_width:ih+$interval_width:$interval_width/2:$interval_width/2:color=$interval_color [v3];
[3:v]setpts=PTS-STARTPTS,pad=iw+$interval_width:ih+$interval_width:$interval_width/2:$interval_width/2:color=$interval_color [v4];
[v1][v2]hstack=inputs=2 [top];  
[v3][v4]hstack=inputs=2 [bottom];
[top][bottom]vstack=inputs=2 [out]"

# 拼接命令
ffmpeg -i "$v1" -i "$v2" -i "$v3" -i "$v4" -filter_complex "$filter_complex" -map "[out]" "$output_video"  -y

echo "Done"
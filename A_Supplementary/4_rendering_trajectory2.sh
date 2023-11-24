
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=5

# crop=-----1368A:912:2742:2
# "crop=342A:228:690:2"


# # # ## ours
/home/yuqi/anaconda3/envs/gpnerf/bin/python gp_nerf/eval.py --val_type=train_instance --supp_name=render_supp_tra  --val_scale_factor=16 --cached_centroids_path='zyq/1115_cluster_yingrenshi/mean_cross_view/mean_depth_200000_1050/test_centroids.npy' --exp_name A_Supplementary_trajectory/yingrenshi --enable_semantic True --network_type gpnerf_nr3d --config_file configs/yingrenshi.yaml --dataset_path /data/yuqi/Datasets/DJI/Yingrenshi_20230926 --batch_size 8192 --train_iterations 200000 --val_interval 20000 --ckpt_interval 20000 --dataset_type memory_depth_dji_instance_crossview --use_scaling False --sampling_mesh_guidance True --sdf_as_gpnerf True --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True --ckpt_path=logs_dji/1018_yingrenshi_density_depth_hash22_far0.3_car2/0/continue150k/0/models/160000.pt --wgt_sem_loss=1 --separate_semantic=True --label_name=1018_ml_fusion_0.3 --num_layers_semantic_hidden=3 --semantic_layer_dim=128 --use_subset=True --lr=0.01 --balance_weight=True --num_semantic_classes=5 --enable_instance=True --freeze_semantic=True --instance_name=instances_mask_0.001_depth --instance_loss_mode=slow_fast --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/1104_yingrenshi_density_depth_hash22_instance_origin_sam_0.001_depth_crossview/0/models/200000.pt


#!/bin/bash
# 设置A文件夹路径
input_folder=A_Supplementary_trajectory/yingrenshi/29
output_video=A_Supplementary_trajectory/29.mp4


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
        ffmpeg -framerate 24 -pattern_type glob -i "$panoptic_folder/*.jpg" -c:v libx264 -b:v 20M -pix_fmt yuv420p -vf "crop=342:228:690:2"  "$output_video_path" -y
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
        ffmpeg -framerate 24 -pattern_type glob -i "$panoptic_folder/*.jpg" -c:v libx264 -b:v 20M -pix_fmt yuv420p -vf "crop=342:228:690:2"  "$output_video_path" -y
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
        ffmpeg -framerate 24 -pattern_type glob -i "$panoptic_folder/*.jpg" -c:v libx264 -b:v 20M -pix_fmt yuv420p -vf "crop=342:228:690:2"  "$output_video_path" -y
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
        ffmpeg -framerate 24 -pattern_type glob -i "$panoptic_folder/*.jpg" -c:v libx264 -b:v 20M -pix_fmt yuv420p -vf "crop=342:228:690:2"  "$output_video_path" -y
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
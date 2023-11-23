
# 输入视频
v1=/data/yuqi/code/GP-NeRF-semantic/A_Supplementary/semantic_video/rgb_b2.mp4
v2=/data/yuqi/code/GP-NeRF-semantic/A_Supplementary/semantic_video/depth_b2.mp4
v3=/data/yuqi/code/GP-NeRF-semantic/A_Supplementary/semantic_video/ours_b2.mp4
v4=/data/yuqi/code/GP-NeRF-semantic/A_Supplementary/instance_video/c_instance_b2.mp4

# 输出视频
output_video=/data/yuqi/code/GP-NeRF-semantic/A_Supplementary/all/all_b2.mp4

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
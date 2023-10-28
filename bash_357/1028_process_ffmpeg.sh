


save_dir=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1028_lh_block1_far0.3

f1=$save_dir/1_labels_m2f_only_sam/labels_merge_vis_alpha
# f2=$save_dir/2_project_to_ori_gt_only_sam/alpha
# f3=$save_dir/3_merge_project_and_sam/labels_merge_vis_alpha
# f4=$save_dir/4_replace_building/alpha
# f5=$save_dir/5_replace_building_cartree/alpha
# f6=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1022_lh_block1_far0.3/project_to_ori_gt_only_sam/project_far_to_ori_replace_building_car_tree/alpha



s1=$save_dir/1
s2=$save_dir/2
s3=$save_dir/3
# s4=$save_dir/4
# s5=$save_dir/5
# s6=$save_dir/6


ffmpeg -r 1 -pattern_type glob -i "${f1}/*.jpg" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p $s1.mp4    -y
# ffmpeg -r 1 -pattern_type glob -i "${f2}/*.jpg" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p $s2.mp4    -y
# ffmpeg -r 1 -pattern_type glob -i "${f3}/*.jpg" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p $s3.mp4    -y
# ffmpeg -r 1 -pattern_type glob -i "${f4}/*.jpg" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p $s4.mp4    -y
# ffmpeg -r 1 -pattern_type glob -i "${f5}/*.jpg" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p $s5.mp4    -y
# ffmpeg -r 1 -pattern_type glob -i "${f6}/*.jpg" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p $s6.mp4    -y


ffmpeg -i $s1.mp4 -i $s2.mp4 -filter_complex hstack $save_dir/temp1.mp4 -y  
ffmpeg -i $save_dir/temp1.mp4 -i $s3.mp4 -filter_complex hstack $save_dir/temp2.mp4 -y

# ffmpeg -i $s4.mp4 -i $s5.mp4 -filter_complex hstack $save_dir/temp3.mp4 -y
# ffmpeg -i $save_dir/temp3.mp4 -i $s6.mp4 -filter_complex hstack $save_dir/temp4.mp4 -y


ffmpeg -i $save_dir/temp2.mp4 -i $save_dir/temp4.mp4 -filter_complex vstack $save_dir/compare.mp4   -y

# rm -rf $save_dir/temp1.mp4   $save_dir/temp2.mp4  $save_dir/temp3.mp4  $save_dir/temp4.mp4   $s1.mp4   $s2.mp4  $s3.mp4  $s4.mp4   $s5.mp4  $s6.mp4



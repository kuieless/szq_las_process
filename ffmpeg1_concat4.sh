
exp_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/0913_demo_idr_depth_semantic/6/eval_200000/val_rgbs
echo $exp_path
cd $exp_path

save_dir=../video
mkdir -p  $save_dir

# ffmpeg -r 1 -pattern_type glob -i "m2f_label/*.jpg" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p $s0.mp4

ffmpeg -r 12 -pattern_type glob -i "pred_rgb/*.jpg"  -vcodec libx264 -crf 18 -pix_fmt yuv420p $save_dir/pred_rgb.mp4   -y
ffmpeg -r 12 -pattern_type glob -i "pred_depth/*.jpg"  -vcodec libx264 -crf 18 -pix_fmt yuv420p $save_dir/pred_depth.mp4   -y
ffmpeg -r 12 -pattern_type glob -i "pred_label/*.jpg"  -vcodec libx264 -crf 18 -pix_fmt yuv420p $save_dir/pred_label.mp4  -y
ffmpeg -r 12 -pattern_type glob -i "m2f_label/*.jpg"  -vcodec libx264 -crf 18 -pix_fmt yuv420p $save_dir/m2f_label.mp4  -y

ffmpeg -r 12 -pattern_type glob -i "*_all.jpg"  -vcodec libx264 -crf 18 -pix_fmt yuv420p $save_dir/all.mp4  -y



ffmpeg -i $save_dir/pred_rgb.mp4 -i $save_dir/pred_depth.mp4 -filter_complex hstack $save_dir/temp1.mp4  -y
ffmpeg -i $save_dir/pred_label.mp4 -i $save_dir/m2f_label.mp4 -filter_complex hstack $save_dir/temp2.mp4  -y

ffmpeg -i $save_dir/temp1.mp4 -i $save_dir/temp2.mp4 -filter_complex vstack $save_dir/concat4.mp4  -y

rm -rf $save_dir/temp1.mp4   $save_dir/temp2.mp4



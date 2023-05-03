
dataset1='Mill19'  #  "Mill19" "   "UrbanScene3D"
dataset2='building' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"
img_root=/data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels
save_dir=./video
mkdir -p  $save_dir

# s0=$save_dir/$dataset2-val
# ffmpeg -r 1 -pattern_type glob -i "${img_root}/val/labels_gt/*.png" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p $s0.mp4
# ffmpeg -r 1 -pattern_type glob -i "${img_root}/val/rgbs/*.JPG" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p $s0.mp4



f0=logs_357/0501_A2_m2f_density_building_separate/1/eval_100000/val_rgbs
s0=$save_dir/$dataset2-val
ffmpeg -r 1 -pattern_type glob -i "${f0}/*_pred_rgb.jpg" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p $s0.mp4


f1=logs_357/0501_A2_m2f_density_building_separate/1/eval_100000/val_rgbs
f2=logs_357/0502_B1_m2f_building_PE_6/0/eval_100000/val_rgbs
f3=logs_357/0502_B2_m2f_building_PE_10/0/eval_100000/val_rgbs
s1=$save_dir/$dataset2-PE2
s2=$save_dir/$dataset2-PE6
s3=$save_dir/$dataset2-PE10

ffmpeg -r 1 -pattern_type glob -i "${f1}/*_pred_label.jpg" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p $s1.mp4
ffmpeg -r 1 -pattern_type glob -i "${f2}/*_pred_label.jpg" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p $s2.mp4
ffmpeg -r 1 -pattern_type glob -i "${f3}/*_pred_label.jpg" -filter:v scale=720:-1  -vcodec libx264 -crf 18 -pix_fmt yuv420p $s3.mp4

ffmpeg -i $s0.mp4 -i $s1.mp4 -filter_complex hstack $save_dir/temp1.mp4
ffmpeg -i $s2.mp4 -i $s3.mp4 -filter_complex hstack $save_dir/temp2.mp4

ffmpeg -i $save_dir/temp1.mp4 -i $save_dir/temp2.mp4 -filter_complex vstack $save_dir/$dataset2-PE.mp4

rm -rf $save_dir/temp1.mp4   $save_dir/temp2.mp4



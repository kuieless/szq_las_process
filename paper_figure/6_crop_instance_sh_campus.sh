
output_path=/data/yuqi/code/GP-NeRF-semantic/paper_figure/4_instance/campus


python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance_GT.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1117_cluster_campus/mean_detectron/mean_depth_200000_1000 \
    --output_path $output_path \
    --custom_name campus_GT



python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1117_cluster_campus/mean_detectron/mean_depth_200000_1000 \
    --output_path $output_path \
    --custom_name campus_CT_detectron



python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1117_cluster_campus/mean_cross_view_all/mean_depth_200000_1000 \
    --output_path $output_path \
    --custom_name campus_CT_crossview_all


python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1117_cluster_campus/mean_sam/mean_depth_200000_1050 \
    --output_path $output_path \
    --custom_name campus_CT_sam




python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/logs_instance/1109_campus_density_depth_hash22_instance_origin_sam_0.001_linear_assignment_crossview/0/eval_100000/panoptic \
    --output_path $output_path \
    --custom_name campus_LA_crossview


python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/logs_instance/1109_campus_density_depth_hash22_instance_origin_sam_0.001_linear_assignment/0/eval_100000/panoptic \
    --output_path $output_path \
    --custom_name campus_LA_sam


python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/logs_instance/1111_campus_detectron_linear_assignment/0/eval_100000/panoptic \
    --output_path $output_path \
    --custom_name campus_LA_detectron




python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_find_images.py  \
    --input_folder_path /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_instance/campus  \
    --target_digits 000010
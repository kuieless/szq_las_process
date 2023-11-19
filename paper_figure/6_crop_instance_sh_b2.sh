
output_path=/data/yuqi/code/GP-NeRF-semantic/paper_figure/4_instance/b2


python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance_GT.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1117_cluster_b2/mean_sam/mean_depth_200000_1000 \
    --output_path $output_path \
    --custom_name b2_GT



python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1117_cluster_b2/mean_sam/mean_depth_200000_1000 \
    --output_path $output_path \
    --custom_name b2_CT_sam



python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1117_cluster_b2/mean_cross_view/mean_depth_200000_1050 \
    --output_path $output_path \
    --custom_name b2_CT_crossview




python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1117_cluster_b2/mean_detectron/mean_depth_200000_1050 \
    --output_path $output_path \
    --custom_name b2_CT_detectron






python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/logs_instance/1109_longhua_b2_density_depth_hash22_instance_origin_sam_0.001_linear_assignment/0/eval_100000/panoptic \
    --output_path $output_path \
    --custom_name b2_LA_sam



python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/logs_instance/1113_b2_density_depth_hash22_instance_origin_sam_0.001_depth_crossview_all_linear_assignment/0/eval_100000/panoptic \
    --output_path $output_path \
    --custom_name b2_LA_crossview_all




python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/logs_instance/1111_b2_detectron_linear_assignment/0/eval_100000/panoptic \
    --output_path $output_path \
    --custom_name b2_LA_detectron



python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_find_images.py  \
    --input_folder_path /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_instance/b2  \
    --target_digits 000002

output_path=/data/yuqi/code/GP-NeRF-semantic/paper_figure/4_instance/yingrenshi


python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance_GT.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1117_cluster_yingrenshi/mean_sam/mean_depth_200000_1050 \
    --output_path $output_path \
    --custom_name yingrenshi_GT



python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1117_cluster_yingrenshi/mean_sam/mean_depth_200000_1050 \
    --output_path $output_path \
    --custom_name yingrenshi_CT_sam



python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1117_cluster_yingrenshi/mean_cross_view/mean_depth_200000_1050 \
    --output_path $output_path \
    --custom_name yingrenshi_CT_crossview




python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1117_cluster_yingrenshi/mean_detectron/mean_depth_200000_1050 \
    --output_path $output_path \
    --custom_name yingrenshi_CT_detectron



# python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
#     --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1117_cluster_yingrenshi/mean_ablation/mean_depth_200000_1050 \
#     --output_path $output_path \
#     --custom_name yingrenshi_CT_ablation





python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/logs_instance/1106_yingrenshi_density_depth_hash22_instance_origin_sam_0.001_linear_assignment/0/eval_200000/panoptic \
    --output_path $output_path \
    --custom_name yingrenshi_LA_sam



# python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
#     --input_path /data/yuqi/code/GP-NeRF-semantic/logs_instance/1104_yingrenshi_density_depth_hash22_instance_origin_sam_0.001_depth_crossview_linear_assignment/0/eval_200000/panoptic \
#     --output_path $output_path \
#     --custom_name yingrenshi_LA_crossview


python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/logs_instance/1106_yingrenshi_density_depth_hash22_instance_origin_sam_0.001_linear_assignment_depth_crossview/0/eval_200000/panoptic \
    --output_path $output_path \
    --custom_name yingrenshi_LA_crossview




python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/logs_instance/1111_yingrenshi_detectron_linear_assignment/0/eval_100000/panoptic \
    --output_path $output_path \
    --custom_name yingrenshi_LA_detectron



python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_find_images.py  \
    --input_folder_path /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_instance/yingrenshi  \
    --target_digits 000001
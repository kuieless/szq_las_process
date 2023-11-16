
output_path=/data/yuqi/code/GP-NeRF-semantic/paper_figure/4_instance/b1


python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance_GT.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1114_cluster_b1/mean_sam/mean_depth_200000_1050 \
    --output_path $output_path \
    --custom_name b1_GT



python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1114_cluster_b1/mean_sam/mean_depth_200000_1050 \
    --output_path $output_path \
    --custom_name b1_CT_sam



python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1114_cluster_b1/mean_cross_view_all/mean_depth_200000_1000 \
    --output_path $output_path \
    --custom_name b1_CT_crossview_all




python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1114_cluster_b1/mean_detectron/mean_depth_200000_1000 \
    --output_path $output_path \
    --custom_name b1_CT_detectron






python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/logs_longhua_b1/1107_longhua_b1_density_depth_hash22_instance_origin_sam_0.001/2/eval_100000/panoptic \
    --output_path $output_path \
    --custom_name b1_LA_sam



python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/logs_longhua_b1/1108_longhua_b1_density_depth_hash22_instance_origin_sam_0.001_depth_crossview_all/4/eval_100000/panoptic \
    --output_path $output_path \
    --custom_name b1_LA_crossview_all




python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/logs_longhua_b1/1111_b1_detectron/1/eval_100000/panoptic \
    --output_path $output_path \
    --custom_name b1_LA_detectron



python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_find_images.py  \
    --input_folder_path /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_instance/b1  \
    --target_digits 000002
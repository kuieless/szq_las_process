
output_path=/data/yuqi/code/GP-NeRF-semantic/paper_figure/4_instance/b1



python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_crop_instance.py \
    --input_path /data/yuqi/code/GP-NeRF-semantic/zyq/1117_cluster_b1/mean_detectron/mean_depth_200000_1000 \
    --output_path $output_path \
    --custom_name b1_CT_detectron





python /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_find_images.py  \
    --input_folder_path /data/yuqi/code/GP-NeRF-semantic/paper_figure/4_instance/b1  \
    --target_digits 000002
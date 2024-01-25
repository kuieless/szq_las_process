#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=2


# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1103_get_sammask_autogenerate_depth_filter.py  \
#     --output_path=logs_rebuttal/get_instance_mask_train_yingrenshi_depth15  \
#     --threshold=0.001   \
#     --dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926  \
#     --geo_filter_long=5


# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1103_get_sammask_autogenerate_depth_filter.py  \
#     --output_path=zyq/1103_get_instance_mask_val_yingrenshi_depth  \
#     --threshold=0.001   \
#     --dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926  \
#     --eval=True




python gp_nerf/train.py \
    --config_file configs/yingrenshi.yaml \
    --dataset_path /data/yuqi/Datasets/DJI/Yingrenshi_20230926 \
    --exp_name ./logs_357/test \
    --batch_size 4096 \
    --label_name=1018_ml_fusion_0.3 \
    --val_interval 1000 \
    --sampling_mesh_guidance True \
    --network_type gpnerf_nr3d \
    --sdf_as_gpnerf=True --freeze_geo=True \
    --log2_hashmap_size=22 \
    --desired_resolution=8192 \
    --ckpt_path=logs_dji/1018_yingrenshi_density_depth_hash22_far0.3_car2/0/continue150k/0/models/160000.pt \
    --use_subset=True \
    --enable_semantic True \
    --dataset_type memory_depth_dji_instance_crossview_process \
    --enable_instance=True \
    --freeze_semantic=True --num_layers_semantic_hidden=3 \
    --instance_name=instances_mask_0.001 \
    --crossview_process_path=0124_rebuttal_get_instance_mask_train_depth_wo_geofilter


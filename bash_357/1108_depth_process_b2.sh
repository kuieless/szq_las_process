#!/bin/bash
export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=7


python gp_nerf/train.py \
    --config_file configs/longhua.yaml \
    --dataset_path /data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds \
    --exp_name ./logs_357/test \
    --batch_size 4096 \
    --label_name=1028_ml_fusion_0.3 \
    --val_interval 1000 \
    --sampling_mesh_guidance True \
    --network_type gpnerf_nr3d \
    --sdf_as_gpnerf=True --freeze_geo=True \
    --log2_hashmap_size=22 \
    --desired_resolution=8192 \
    --ckpt_path=logs_longhua_b1/1029_longhua_b1_density_depth_hash22_car2_semantic_1028_fusion/0/models/200000.pt \
    --use_subset=True \
    --enable_semantic True \
    --dataset_type memory_depth_dji_instance_crossview_process \
    --enable_instance=True \
    --freeze_semantic=True --num_layers_semantic_hidden=3 \
    --instance_name=instances_mask_0.001_depth \
    --train_scale_factor=1 \
    --val_scale_factor=1 \
    --crossview_process_path=1108_get_instance_mask_train_longhua_b1_depth_0.001_process
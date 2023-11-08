#!/bin/bash
export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=5


python gp_nerf/train.py \
    --config_file configs/campus_new.yaml \
    --dataset_path /data/yuqi/Datasets/DJI/Campus_new \
    --exp_name ./logs_357/test \
    --batch_size 4096 \
    --label_name=1018_ml_fusion_0.3 \
    --val_interval 1000 \
    --sampling_mesh_guidance True \
    --network_type gpnerf_nr3d \
    --sdf_as_gpnerf=True --freeze_geo=True \
    --log2_hashmap_size=22 \
    --desired_resolution=8192 \
    --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_campus/1104_campus_density_depth_hash22_fusion1018_car2_semantic/0/models/200000.pt \
    --use_subset=True \
    --enable_semantic True \
    --dataset_type memory_depth_dji_instance_crossview_process \
    --enable_instance=True \
    --freeze_semantic=True --num_layers_semantic_hidden=3 \
    --instance_name=instances_mask_0.001_depth \
    --crossview_process_path=1108_get_instance_mask_train_campus_depth_0.001_process
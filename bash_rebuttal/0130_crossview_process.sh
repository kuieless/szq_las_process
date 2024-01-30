#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1





python gp_nerf/train.py \
    --config_file configs/instancebuilding1_jx.yaml \
    --dataset_path /data/yuqi/Datasets/InstanceBuilding/3D/scene1/colmap_jx_new \
    --exp_name ./logs_357/test \
    --batch_size 4096 \
    --label_name=gt \
    --val_interval 1000 \
    --sampling_mesh_guidance False \
    --network_type gpnerf_nr3d \
    --sdf_as_gpnerf=True --freeze_geo=True \
    --log2_hashmap_size=22 \
    --desired_resolution=8192 \
    --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/0129_ib1_geo/0/models/140000.pt \
    --use_subset=True \
    --enable_semantic True \
    --dataset_type memory_depth_dji_instance_crossview_process \
    --enable_instance=True \
    --freeze_semantic=True --num_layers_semantic_hidden=3 \
    --instance_name=instances_mask_0.001_depth20 \
    --crossview_process_path=0130_IB_instance_mask_0.001_process_depth20


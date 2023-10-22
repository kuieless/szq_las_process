#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5
export CUDA_LAUNCH_BLOCKING=1
python gp_nerf/eval.py --exp_name logs_dji/1021_lh_block1_density_depth_hash22_ds_cropdepth --enable_semantic False --network_type gpnerf_nr3d --config_file configs/longhua.yaml --dataset_path /data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds --batch_size 10240 --train_iterations 200000 --val_interval 50000 --ckpt_interval 50000 --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance True --sdf_as_gpnerf True --geo_init_method=idr --depth_dji_loss True --wgt_depth_mse_loss 1 --wgt_sigma_loss 0 --log2_hashmap_size=22 --desired_resolution=8192 --train_scale_factor=1 --val_scale_factor=1  --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/1021_lh_block1_density_depth_hash22_ds_cropdepth/0/models/200000.pt  --render_zyq


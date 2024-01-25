#!/bin/bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=6

python gp_nerf/eval.py --exp_name logs_demo/0120_cuhksz_demo_geo_renderfar   --render_zyq  --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_demo/0120_cuhksz_demo_geo/0/models/300000.pt --enable_semantic False --network_type gpnerf_nr3d --config_file configs/cuhksz.yaml --batch_size 10240 --train_iterations 300000 --val_interval 50000 --ckpt_interval 50000 --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance True --sdf_as_gpnerf True --geo_init_method=idr --depth_dji_loss True --wgt_depth_mse_loss 1 --wgt_sigma_loss 0

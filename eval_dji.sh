#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

exp_name=logs_dji/0802_2_G_DJI
ckpt_path=logs_dji/0802_2_G_DJI/0/models/50000.pt
enable_semantic=False

python gp_nerf/eval.py   --val_type  'val'   --config_file  dji/cuhksz_ray_xml.yaml   \
    --dataset_path  /data/yuqi/Datasets/DJI/DJI_20230726_xiayuan    \
    --exp_name  $exp_name    --ckpt_path  $ckpt_path    --enable_semantic  $enable_semantic  \
    --depth_dji_loss True














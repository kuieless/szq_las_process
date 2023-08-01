#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7


python scripts/save_depth.py  --exp_name   ./logs_357/test   \
    --dataset_path   /data/yuqi/Datasets/DJI/DJI_20230726_xiayuan    --config_file  dji/cuhksz_ray_xml.yaml      \
    --ckpt_path  logs_357/0729_G_DJI/0/models/200000.pt  --dataset_type  memory









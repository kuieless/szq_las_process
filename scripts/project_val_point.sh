#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1


python scripts/project_val_point.py  --exp_name   ./logs_357/0510_project_point   --dataset_path   /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels    --config_file  configs/residence.yaml  
#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=2

exp_name='logs_313/0430_Z2_m2f_density_building_largeMLP_separate'

dataset1='Mill19'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='building' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"
stop_semantic_grad=True
use_pano_lift=False

batch_size=30720
train_iterations=300000
val_interval=50000
ckpt_interval=50000
wandb_id=gpnerf_semantic  #gpnerf_semantic   None
wandb_run_name=$exp_name


network_type='gpnerf'     #  gpnerf   sdf
ignore_index=-1
label_name='m2f'      # merge    m2f
separate_semantic=True  # True False

python gp_nerf/train.py  --exp_name  $exp_name  --wandb_id  $wandb_id  --label_name   $label_name   --separate_semantic   $separate_semantic   --ignore_index  $ignore_index  --network_type    $network_type   --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_name-down4      --train_iterations   $train_iterations        --val_interval  $val_interval    --ckpt_interval   $ckpt_interval      --stop_semantic_grad  $stop_semantic_grad  --use_pano_lift  $use_pano_lift     --wandb_run_name  $wandb_run_name    --batch_size  $batch_size   










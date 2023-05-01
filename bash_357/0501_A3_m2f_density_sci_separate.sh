#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4

exp_name='logs_357/0501_A3_m2f_density_sci_separate'
dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='sci-art' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"
wandb_id=gpnerf_semantic  #gpnerf_semantic   None


wandb_run_name=$exp_name
separate_semantic=True  # True False
network_type='gpnerf'     #  gpnerf   sdf
label_name='m2f'      # merge    m2f


batch_size=40960
train_iterations=200000
val_interval=50000
ckpt_interval=50000



python gp_nerf/train.py  --wandb_id  $wandb_id   --exp_name  $exp_name    --label_name   $label_name   --separate_semantic   $separate_semantic    --network_type    $network_type   --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_name-down4      --train_iterations   $train_iterations        --val_interval  $val_interval    --ckpt_interval   $ckpt_interval      --wandb_run_name  $wandb_run_name    --batch_size  $batch_size   










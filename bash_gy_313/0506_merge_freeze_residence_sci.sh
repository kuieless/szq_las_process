#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

exp_name='logs_gy_313/0506_merge_freeze_residence'
dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='residence' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"
wandb_id=None #gpnerf_semantic  #gpnerf_semantic   None

wandb_run_name=$exp_name
separate_semantic=True  # True False
network_type='gpnerf'     #  gpnerf   sdf
label_name='merge_new'      # merge_new    m2f

batch_size=40960
train_iterations=100000
val_interval=5000
ckpt_interval=5000
pos_xyz_dim=10

enable_semantic=True
freeze_geo=True
ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_357/0504_G3_geo_residence/0/models/200000.pt
python gp_nerf/train.py  --wandb_id  $wandb_id   --exp_name  $exp_name  --ckpt_path  $ckpt_path  --freeze_geo $freeze_geo --enable_semantic  $enable_semantic --pos_xyz_dim    $pos_xyz_dim  --label_name   $label_name   --separate_semantic   $separate_semantic    --network_type    $network_type   --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_name-down4      --train_iterations   $train_iterations        --val_interval  $val_interval    --ckpt_interval   $ckpt_interval      --wandb_run_name  $wandb_run_name    --batch_size  $batch_size   



exp_name='logs_357/0505_D3_merge_freeze_sci'
dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='sci-art' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"

wandb_run_name=$exp_name
separate_semantic=True  # True False
network_type='gpnerf'     #  gpnerf   sdf
label_name='merge_new'      # merge    m2f

batch_size=40960
train_iterations=100000
val_interval=50000
ckpt_interval=50000
pos_xyz_dim=10

enable_semantic=True
freeze_geo=True
ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_357/0504_G4_geo_sci/0/models/200000.pt
python gp_nerf/train.py  --wandb_id  $wandb_id   --exp_name  $exp_name  --ckpt_path  $ckpt_path  --freeze_geo $freeze_geo --enable_semantic  $enable_semantic --pos_xyz_dim    $pos_xyz_dim  --label_name   $label_name   --separate_semantic   $separate_semantic    --network_type    $network_type   --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_name-down4      --train_iterations   $train_iterations        --val_interval  $val_interval    --ckpt_interval   $ckpt_interval      --wandb_run_name  $wandb_run_name    --batch_size  $batch_size   








#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5

####  1104_campus_render_far.sh

# python gp_nerf/eval.py \
#     --exp_name logs_rebuttal/0125_campus_farview_1.0_val \
#     --enable_semantic False \
#     --network_type gpnerf_nr3d \
#     --config_file configs/campus_new.yaml \
#     --dataset_path /data/yuqi/Datasets/DJI/Campus_new \
#     --batch_size 10240 --train_iterations 200000 --val_interval 50000 --ckpt_interval 50000 \
#     --dataset_type memory_depth_dji --use_scaling False \
#     --sampling_mesh_guidance True --sdf_as_gpnerf True \
#     --geo_init_method=idr --depth_dji_loss True \
#     --wgt_depth_mse_loss 1 --wgt_sigma_loss 0 \
#     --log2_hashmap_size=22 --desired_resolution=8192 \
#     --ckpt_path=logs_campus/1103_campus_density_depth_hash22/0/models/200000.pt \
#     --render_zyq



output=/data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/0125_campus_farview_1.0_val/mask2former_output

/data/yuqi/anaconda3/envs/mask2former/bin/python /data/yuqi/code/Mask2Former_a/demo/demo_zyq_0502_augment.py  \
    --config-file  /data/yuqi/code/Mask2Former_a/configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml \
    --input  /data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/0125_campus_farview_1.0_val/pred_rgb  \
    --output  $output \
    --zyq_code  True  \
    --zyq_mapping True \
    --opts MODEL.WEIGHTS  /data/yuqi/code/Mask2Former_a/pretrained_ckpt/ade20k/model_final_e0c58e_panoptic_swinL.pkl

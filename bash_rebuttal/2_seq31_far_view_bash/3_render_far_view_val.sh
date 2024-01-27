export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=3

python  gp_nerf/eval.py \
    --exp_name logs_rebuttal/0127_seq31_geo_farview0.2_val_agu \
    --enable_semantic False --network_type gpnerf_nr3d \
    --config_file configs/seq31.yaml \
    --dataset_path /data/yuqi/Datasets/DJI/uavid/seq31 \
    --batch_size 10240 --train_iterations 200000 --val_interval 20000 --ckpt_interval 20000 \
    --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance False --sdf_as_gpnerf True \
    --geo_init_method=idr --depth_dji_loss False --wgt_depth_mse_loss 0 --wgt_sigma_loss 0 \
    --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/0127_seq31_geo/0/models/60000.pt \
    --render_zyq --render_zyq_far_view=render_far0.2_val


cd /data/yuqi/code/Mask2Former_a

output=/data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/0127_seq31_geo_farview0.2_val_agu/0/eval_60000/val_rgbs

/data/yuqi/anaconda3/envs/mask2former/bin/python demo/demo_zyq_0502_augment.py  \
    --config-file  configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml \
    --input  $output/pred_rgb  \
    --output  $output/mask2former_output \
    --zyq_code  True  \
    --zyq_mapping True \
    --opts MODEL.WEIGHTS  pretrained_ckpt/ade20k/model_final_e0c58e_panoptic_swinL.pkl

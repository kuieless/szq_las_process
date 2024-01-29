export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=7


python  gp_nerf/eval.py \
    --enable_semantic True --network_type gpnerf_nr3d \
    --config_file configs/seq31.yaml \
    --dataset_path /data/yuqi/Datasets/DJI/uavid/seq31 \
    --batch_size 10240 --train_iterations 200000 --val_interval 20000 --ckpt_interval 20000 \
    --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance False --sdf_as_gpnerf True \
    --geo_init_method=idr --depth_dji_loss False --wgt_depth_mse_loss 0 --wgt_sigma_loss 0 \
    --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/0127_seq31_geo/0/models/60000.pt \
    --eval_others=True \
    --eval_others_name=labels_0125_rebu_0.2  \
    --exp_name logs_rebuttal/0127_seq31_eval_2d_ours_0.2_remapping



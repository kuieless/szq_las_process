export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=6




python  gp_nerf/eval.py \
    --exp_name logs_rebuttal/0127_seq31_geo_sem_m2f_remapping \
    --enable_semantic True --network_type gpnerf_nr3d --config_file configs/seq31.yaml --dataset_path /data/yuqi/Datasets/DJI/uavid/seq31 \
    --batch_size 20480 --train_iterations 200000 --val_interval 20000 --ckpt_interval 20000 --dataset_type memory_depth_dji --use_scaling False \
    --sampling_mesh_guidance False --sdf_as_gpnerf True --geo_init_method=idr --depth_dji_loss False --wgt_depth_mse_loss 0 --wgt_sigma_loss 0 \
    --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True \
    --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/2_seq31/0127_seq31_geo_sem_m2f/0/models/200000.pt \
    --wgt_sem_loss=1 --separate_semantic=True --label_name=m2f --num_layers_semantic_hidden=3 --semantic_layer_dim=128 --lr=0.01 --balance_weight=True \
    --num_semantic_classes=5



export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0

## 得到train/val的depth，供后面投影使用

python  gp_nerf/eval.py \
    --exp_name logs_rebuttal/0126_seq14_geo_savedepth \
    --enable_semantic False --network_type gpnerf_nr3d \
    --config_file configs/seq14.yaml \
    --dataset_path /data/yuqi/jx_rebuttal/seq14_ds_10_val/postprocess_zyq \
    --batch_size 10240 --train_iterations 200000 --val_interval 20000 --ckpt_interval 20000 \
    --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance False --sdf_as_gpnerf True \
    --geo_init_method=idr --depth_dji_loss False --wgt_depth_mse_loss 0 --wgt_sigma_loss 0 \
    --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/0125_seq14_geo/4/models/40000.pt \
    --save_depth=True   \
    # --val_type=train_save_depth    ## 训练视角加了这一行


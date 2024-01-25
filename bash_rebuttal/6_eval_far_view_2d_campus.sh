export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=5


python gp_nerf/eval.py \
    --enable_semantic True --network_type gpnerf_nr3d --config_file configs/campus_new.yaml \
    --dataset_path /data/yuqi/Datasets/DJI/Campus_new \
    --batch_size 40960 --train_iterations 200000 --val_interval 20000 --ckpt_interval 20000 \
    --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance True \
    --sdf_as_gpnerf True --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True \
    --ckpt_path=logs_campus/1103_campus_density_depth_hash22/0/models/200000.pt --wgt_sem_loss=1 \
    --separate_semantic=True --label_name=m2f --num_layers_semantic_hidden=3 \
    --semantic_layer_dim=128 --use_subset=True --lr=0.01 --balance_weight=True --num_semantic_classes=5 \
    --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_campus/1104_campus_density_depth_hash22_m2f_car2_semantic/0/models/200000.pt \
    --eval_others=True \
    --eval_others_name=labels_0125_rebu_0.5    \
    --exp_name logs_rebuttal/0125_campus_farview_0.5_val_metric 

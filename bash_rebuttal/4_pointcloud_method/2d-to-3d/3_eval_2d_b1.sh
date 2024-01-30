export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=5



python gp_nerf/eval.py \
    --enable_semantic True --network_type gpnerf_nr3d \
    --batch_size 40960 --train_iterations 200000 --val_interval 10000 --ckpt_interval 10000 \
    --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance True \
    --sdf_as_gpnerf True --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True \
    --ckpt_path=logs_dji/1003_b1_density_depth_hash22/0/models/200000.pt --wgt_sem_loss=1 \
    --separate_semantic=True --label_name=1018_ml_fusion_0.3 --num_layers_semantic_hidden=3 \
    --semantic_layer_dim=128 --use_subset=True --lr=0.01 --balance_weight=True --num_semantic_classes=5 \
    --config_file configs/longhua1.yaml \
    --dataset_path /data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds \
    --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_longhua_b1/1029_longhua_b1_density_depth_hash22_car2_semantic_1028_fusion/0/models/140000.pt \
    --eval_others=True \
    --eval_others_name=3d_voting_ours  \
    --exp_name zyq_rebuttal_4/eval_2d/b1_3d_voting_ours



python gp_nerf/eval.py \
    --enable_semantic True --network_type gpnerf_nr3d \
    --batch_size 40960 --train_iterations 200000 --val_interval 10000 --ckpt_interval 10000 \
    --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance True \
    --sdf_as_gpnerf True --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True \
    --ckpt_path=logs_dji/1003_b1_density_depth_hash22/0/models/200000.pt --wgt_sem_loss=1 \
    --separate_semantic=True --label_name=1018_ml_fusion_0.3 --num_layers_semantic_hidden=3 \
    --semantic_layer_dim=128 --use_subset=True --lr=0.01 --balance_weight=True --num_semantic_classes=5 \
    --config_file configs/longhua1.yaml \
    --dataset_path /data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds \
    --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_longhua_b1/1029_longhua_b1_density_depth_hash22_car2_semantic_1028_fusion/0/models/140000.pt \
    --eval_others=True \
    --eval_others_name=3d_voting_m2f  \
    --exp_name zyq_rebuttal_4/eval_2d/b1_3d_voting_m2f



python gp_nerf/eval.py \
    --enable_semantic True --network_type gpnerf_nr3d \
    --batch_size 40960 --train_iterations 200000 --val_interval 10000 --ckpt_interval 10000 \
    --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance True \
    --sdf_as_gpnerf True --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True \
    --ckpt_path=logs_dji/1003_b1_density_depth_hash22/0/models/200000.pt --wgt_sem_loss=1 \
    --separate_semantic=True --label_name=1018_ml_fusion_0.3 --num_layers_semantic_hidden=3 \
    --semantic_layer_dim=128 --use_subset=True --lr=0.01 --balance_weight=True --num_semantic_classes=5 \
    --config_file configs/longhua1.yaml \
    --dataset_path /data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds \
    --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_longhua_b1/1029_longhua_b1_density_depth_hash22_car2_semantic_1028_fusion/0/models/140000.pt \
    --eval_others=True \
    --eval_others_name=3d_results_randlanet  \
    --exp_name zyq_rebuttal_4/eval_2d/b1_3d_results_randlanet
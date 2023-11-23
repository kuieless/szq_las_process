
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=2

# yingrenshi ours
# python gp_nerf/eval.py --val_type=train_instance --exp_name A_Supplementary/rendering_semantic/yingrenshi/1018_yingrenshi_density_depth_hash22_far0.3_car2 --enable_semantic True --network_type gpnerf_nr3d --config_file configs/yingrenshi.yaml --dataset_path /data/yuqi/Datasets/DJI/Yingrenshi_20230926 --batch_size 40960 --train_iterations 200000 --val_interval 10000 --ckpt_interval 10000 --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance True --sdf_as_gpnerf True --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True --ckpt_path=logs_dji/1003_yingrenshi_density_depth_hash22/0/models/200000.pt --wgt_sem_loss=1 --separate_semantic=True --label_name=1018_ml_fusion_0.3 --num_layers_semantic_hidden=3 --semantic_layer_dim=128 --use_subset=True --lr=0.01 --balance_weight=True --num_semantic_classes=5 --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/1018_yingrenshi_density_depth_hash22_far0.3_car2/0/continue150k/0/models/170000.pt

# # yingrenshi m2f
python gp_nerf/eval.py --val_type=train_instance --exp_name A_Supplementary/rendering_semantic/yingrenshi/1013_yingrenshi_density_depth_hash22_m2f --enable_semantic True --network_type gpnerf_nr3d --config_file configs/yingrenshi.yaml --dataset_path /data/yuqi/Datasets/DJI/Yingrenshi_20230926 --batch_size 40960 --train_iterations 200000 --val_interval 10000 --ckpt_interval 10000 --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance True --sdf_as_gpnerf True --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True --wgt_sem_loss=1 --separate_semantic=True --label_name=m2f --num_layers_semantic_hidden=3 --semantic_layer_dim=128 --use_subset=True --lr=0.01 --balance_weight=False --ckpt_path=logs_yingrenshi/1013_yingrenshi_density_depth_hash22_m2f/0/models/200000.pt --num_semantic_classes=11



# # b2 ours 
# python gp_nerf/eval.py --val_type=train_instance --exp_name A_Supplementary/rendering_semantic/longhua_b2/1029_longhua_b2_density_depth_hash22_car2_semantic_1028_fusion --enable_semantic True --network_type gpnerf_nr3d --config_file configs/longhua2.yaml --dataset_path /data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds --batch_size 40960 --train_iterations 200000 --val_interval 20000 --ckpt_interval 20000 --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance True --sdf_as_gpnerf True --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True --ckpt_path=logs_dji/1021_lh_block2_density_depth_hash22_ds_cropdepth/0/models/200000.pt --wgt_sem_loss=1 --separate_semantic=True --label_name=1028_ml_fusion_0.3 --num_layers_semantic_hidden=3 --semantic_layer_dim=128 --use_subset=True --lr=0.01 --balance_weight=True --num_semantic_classes=5 --train_scale_factor=1 --val_scale_factor=1 --ckpt_path=logs_longhua_b2/1029_longhua_b2_density_depth_hash22_car2_semantic_1028_fusion/0/models/200000.pt

# # b2 m2f

python gp_nerf/eval.py --val_type=train_instance --exp_name A_Supplementary/rendering_semantic/longhua_b2/1028_longhua_b2_density_depth_hash22_m2f_car2_semantic --enable_semantic True --network_type gpnerf_nr3d --config_file configs/longhua2.yaml --dataset_path /data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds --batch_size 40960 --train_iterations 200000 --val_interval 10000 --ckpt_interval 10000 --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance True --sdf_as_gpnerf True --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True --ckpt_path=logs_dji/1021_lh_block2_density_depth_hash22_ds_cropdepth/0/models/200000.pt --wgt_sem_loss=1 --separate_semantic=True --label_name=m2f --num_layers_semantic_hidden=3 --semantic_layer_dim=128 --use_subset=True --lr=0.01 --balance_weight=True --num_semantic_classes=5 --train_scale_factor=1 --val_scale_factor=1 --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_longhua_b2/1028_longhua_b2_density_depth_hash22_m2f_car2_semantic/0/models/200000.pt



# b1 ours
# python gp_nerf/eval.py --val_type=train_instance --exp_name A_Supplementary/rendering_semantic/longhua_b1/1029_longhua_b1_density_depth_hash22_car2_semantic_1028_fusion --enable_semantic True --network_type gpnerf_nr3d --config_file configs/longhua1.yaml --dataset_path /data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds --batch_size 40960 --train_iterations 200000 --val_interval 20000 --ckpt_interval 20000 --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance True --sdf_as_gpnerf True --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True --ckpt_path=logs_dji/1021_lh_block1_density_depth_hash22_ds_cropdepth/0/models/200000.pt --wgt_sem_loss=1 --separate_semantic=True --label_name=1028_ml_fusion_0.3 --num_layers_semantic_hidden=3 --semantic_layer_dim=128 --use_subset=True --lr=0.01 --balance_weight=True --num_semantic_classes=5 --train_scale_factor=1 --val_scale_factor=1 --ckpt_path=logs_longhua_b1/1029_longhua_b1_density_depth_hash22_car2_semantic_1028_fusion/0/models/200000.pt


# # b1 m2f
python gp_nerf/eval.py --val_type=train_instance --exp_name A_Supplementary/rendering_semantic/longhua_b1/1028_longhua_b1_density_depth_hash22_m2f_car2_semantic --enable_semantic True --network_type gpnerf_nr3d --config_file configs/longhua1.yaml --dataset_path /data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds --batch_size 40960 --train_iterations 200000 --val_interval 10000 --ckpt_interval 10000 --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance True --sdf_as_gpnerf True --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True --ckpt_path=logs_dji/1021_lh_block1_density_depth_hash22_ds_cropdepth/0/models/200000.pt --wgt_sem_loss=1 --separate_semantic=True --label_name=m2f --num_layers_semantic_hidden=3 --semantic_layer_dim=128 --use_subset=True --lr=0.01 --balance_weight=True --num_semantic_classes=5 --train_scale_factor=1 --val_scale_factor=1 --ckpt_path=logs_longhua_b1/1028_longhua_b1_density_depth_hash22_m2f_car2_semantic/0/models/200000.pt



# # # campus ours 
# # python gp_nerf/eval.py --val_type=train_instance --exp_name A_Supplementary/rendering_semantic/campus/1104_campus_density_depth_hash22_fusion1018_car2_semantic --enable_semantic True --network_type gpnerf_nr3d --config_file configs/campus_new.yaml --dataset_path /data/yuqi/Datasets/DJI/Campus_new --batch_size 40960 --train_iterations 200000 --val_interval 20000 --ckpt_interval 20000 --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance True --sdf_as_gpnerf True --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True --ckpt_path=logs_campus/1103_campus_density_depth_hash22/0/models/200000.pt --wgt_sem_loss=1 --separate_semantic=True --label_name=1018_ml_fusion_0.3 --num_layers_semantic_hidden=3 --semantic_layer_dim=128 --use_subset=True --lr=0.01 --balance_weight=True --num_semantic_classes=5 --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_campus/1104_campus_density_depth_hash22_fusion1018_car2_semantic/0/models/200000.pt

# # # campus m2f
python gp_nerf/eval.py --val_type=train_instance --exp_name A_Supplementary/rendering_semantic/campus/1104_campus_density_depth_hash22_m2f_car2_semantic --enable_semantic True --network_type gpnerf_nr3d --config_file configs/campus_new.yaml --dataset_path /data/yuqi/Datasets/DJI/Campus_new --batch_size 40960 --train_iterations 200000 --val_interval 20000 --ckpt_interval 20000 --dataset_type memory_depth_dji --use_scaling False --sampling_mesh_guidance True --sdf_as_gpnerf True --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True --ckpt_path=logs_campus/1103_campus_density_depth_hash22/0/models/200000.pt --wgt_sem_loss=1 --separate_semantic=True --label_name=m2f --num_layers_semantic_hidden=3 --semantic_layer_dim=128 --use_subset=True --lr=0.01 --balance_weight=True --num_semantic_classes=5 --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_campus/1104_campus_density_depth_hash22_m2f_car2_semantic/0/models/200000.pt







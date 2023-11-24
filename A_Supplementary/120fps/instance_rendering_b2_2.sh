
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=3






# ## b2  LA  sam
# python gp_nerf/eval.py --val_type=train_instance --exp_name A_Supplementary/render_instance/b2/1109_longhua_b2_density_depth_hash22_instance_origin_sam_0.001_linear_assignment --enable_semantic True --network_type gpnerf_nr3d --config_file configs/longhua2.yaml --dataset_path /data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds --batch_size 16384 --train_iterations 100000 --val_interval 20000 --ckpt_interval 20000 --dataset_type memory_depth_dji_instance --use_scaling False --sampling_mesh_guidance True --sdf_as_gpnerf True --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True --ckpt_path=logs_longhua_b2/1029_longhua_b2_density_depth_hash22_car2_semantic_1028_fusion/0/models/200000.pt --wgt_sem_loss=1 --separate_semantic=True --label_name=1028_ml_fusion_0.3 --num_layers_semantic_hidden=3 --semantic_layer_dim=128 --use_subset=True --lr=0.01 --balance_weight=True --num_semantic_classes=5 --enable_instance=True --freeze_semantic=True --instance_name=instances_mask_0.001 --instance_loss_mode=linear_assignment --num_instance_classes=100 --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_longhua_b2/1109_longhua_b2_density_depth_hash22_instance_origin_sam_0.001_linear_assignment/0/models/100000.pt


# ## b2  LA  ours
# python gp_nerf/eval.py --val_type=train_instance --exp_name A_Supplementary/render_instance/b2/1113_b2_density_depth_hash22_instance_origin_sam_0.001_depth_crossview_all_linear_assignment --enable_semantic True --config_file configs/longhua2.yaml --batch_size 16384 --train_iterations 100000 --val_interval 50000 --ckpt_interval 50000 --dataset_type memory_depth_dji_instance_crossview --freeze_geo=True --ckpt_path=logs_longhua_b2/1029_longhua_b2_density_depth_hash22_car2_semantic_1028_fusion/0/models/200000.pt --use_subset=True --lr=0.01 --enable_instance=True --freeze_semantic=True --instance_name=instances_mask_0.001_depth --instance_loss_mode=linear_assignment --num_instance_classes=100 --crossview_all=True --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_longhua_b2/1113_b2_density_depth_hash22_instance_origin_sam_0.001_depth_crossview_all_linear_assignment/0/models/100000.pt


## b2  LA  detectron
python gp_nerf/eval.py --val_type=train_instance --exp_name A_Supplementary/render_instance/b2/1111_b2_detectron_linear_assignment --enable_semantic True --config_file configs/longhua2.yaml --batch_size 16384 --train_iterations 100000 --val_interval 50000 --ckpt_interval 50000 --dataset_type memory_depth_dji_instance --freeze_geo=True --ckpt_path=logs_longhua_b2/1029_longhua_b2_density_depth_hash22_car2_semantic_1028_fusion/0/models/200000.pt --use_subset=True --lr=0.01 --enable_instance=True --freeze_semantic=True --instance_name=instances_detectron --instance_loss_mode=linear_assignment --num_instance_classes=100 --ckpt_path=logs_longhua_b2/1111_b2_detectron_linear_assignment/0/models/100000.pt


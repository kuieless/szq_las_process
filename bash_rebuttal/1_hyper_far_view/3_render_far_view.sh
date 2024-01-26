export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=5

# python gp_nerf/eval.py \
#     --exp_name logs_rebuttal/0125_farview_1.0_val \
#     --enable_semantic True --network_type gpnerf_nr3d \
#     --config_file configs/yingrenshi.yaml \
#     --dataset_path /data/yuqi/Datasets/DJI/Yingrenshi_20230926 \
#     --batch_size 40960 --train_iterations 200000 --val_interval 10000 --ckpt_interval 10000 \
#     --dataset_type memory_depth_dji --use_scaling False \
#     --sampling_mesh_guidance True --sdf_as_gpnerf True \
#     --log2_hashmap_size=22 --desired_resolution=8192 --freeze_geo=True \
#     --ckpt_path=logs_dji/1003_yingrenshi_density_depth_hash22/0/models/200000.pt \
#     --wgt_sem_loss=1 --separate_semantic=True \
#     --label_name=1018_ml_fusion_0.3 \
#     --num_layers_semantic_hidden=3 \
#     --semantic_layer_dim=128 \
#     --use_subset=True --lr=0.01 \
#     --balance_weight=True --num_semantic_classes=5 \
#     --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/1018_yingrenshi_density_depth_hash22_far0.3_car2/0/continue150k/0/models/170000.pt \
#     --render_zyq



output=/data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/0125_farview_1.0_val/mask2former_output

/data/yuqi/anaconda3/envs/mask2former/bin/python demo/demo_zyq_0502_augment.py  \
    --config-file  configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml \
    --input  /data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/0125_farview_1.0_val/pred_rgb  \
    --output  $output \
    --zyq_code  True  \
    --zyq_mapping True \
    --opts MODEL.WEIGHTS  pretrained_ckpt/ade20k/model_final_e0c58e_panoptic_swinL.pkl

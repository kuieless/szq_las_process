
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5

output_path=

# python /data/yuqi/code/GP-NeRF-semantic/gp_nerf/query_nerf_result.py \
#     --config_file=configs/longhua1.yaml   \
#     --output_path=/data/yuqi/code/GP-NeRF-semantic/zyq_rebuttal_4/query_nerf/b1.ply \
#     --dataset_path=/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds \
#     --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_longhua_b1/1029_longhua_b1_density_depth_hash22_car2_semantic_1028_fusion/0/models/200000.pt \
#     --metaXml_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/data/xml/b1.xml \
#     --ply_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/data/gt/gt_label_full_reversed_fineReg_trans.txt \
#     --exp_name=logs_dji/test \
#     --enable_semantic=True


python /data/yuqi/code/GP-NeRF-semantic/gp_nerf/query_nerf_result.py \
    --config_file=configs/longhua1.yaml   \
    --output_path=/data/yuqi/code/GP-NeRF-semantic/zyq_rebuttal_4/query_semantic_nerf/b1.ply \
    --dataset_path=/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds \
    --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_longhua_b1/1028_longhua_b1_density_depth_hash22_m2f_car2_semantic/0/models/200000.pt \
    --metaXml_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/data/xml/b1.xml \
    --ply_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/data/gt/gt_label_full_reversed_fineReg_trans.txt \
    --exp_name=logs_dji/test \
    --enable_semantic=True
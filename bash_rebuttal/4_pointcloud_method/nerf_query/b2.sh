
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6

output_path=

python /data/yuqi/code/GP-NeRF-semantic/gp_nerf/query_nerf_result.py \
    --config_file=configs/longhua2.yaml   \
    --output_path=/data/yuqi/code/GP-NeRF-semantic/zyq_rebuttal_4/query_nerf/b2.ply \
    --dataset_path=/data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds \
    --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_longhua_b2/1029_longhua_b2_density_depth_hash22_car2_semantic_1028_fusion/0/models/200000.pt \
    --metaXml_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/data/xml/b2.xml \
    --ply_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/data/gt/longhua_block2_gt_label_finRegistered.txt \
    --exp_name=logs_dji/test \
    --enable_semantic=True
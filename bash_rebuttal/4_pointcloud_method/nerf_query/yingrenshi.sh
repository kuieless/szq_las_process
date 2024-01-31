
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5



####  ours 

# python /data/yuqi/code/GP-NeRF-semantic/gp_nerf/query_nerf_result.py \
#     --config_file=configs/yingrenshi.yaml   \
#         --output_path=/data/yuqi/code/GP-NeRF-semantic/zyq_rebuttal_4/query_nerf/yingrenshi.ply \
#         --dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926 \
#         --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/1018_yingrenshi_density_depth_hash22_far0.3_car2/0/continue150k/0/models/160000.pt \
#         --metaXml_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/data/xml/yingrenshi.xml \
#         --ply_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/data/gt/Yingrenshi_fine-registered_full.txt \
#         --exp_name=logs_dji/test \
#         --enable_semantic=True 



####  semantic-nerf(M2F)

python /data/yuqi/code/GP-NeRF-semantic/gp_nerf/query_nerf_result.py \
    --config_file=configs/yingrenshi.yaml   \
        --output_path=/data/yuqi/code/GP-NeRF-semantic/zyq_rebuttal_4/query_semantic_nerf/yingrenshi.ply \
        --dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926 \
        --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_yingrenshi/1013_yingrenshi_density_depth_hash22_m2f/0/models/200000.pt \
        --metaXml_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/data/xml/yingrenshi.xml \
        --ply_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/data/gt/Yingrenshi_fine-registered_full.txt \
        --exp_name=logs_dji/test \
        --enable_semantic=True \
        --num_semantic_classes=11




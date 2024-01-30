
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5




python /data/yuqi/code/GP-NeRF-semantic/gp_nerf/query_nerf_result.py \
    --config_file=configs/campus_new.yaml   \
        --output_path=/data/yuqi/code/GP-NeRF-semantic/zyq_rebuttal_4/query_nerf/campus.ply \
        --dataset_path=/data/yuqi/Datasets/DJI/Campus_new \
        --ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_campus/1104_campus_density_depth_hash22_fusion1018_car2_semantic/0/models/200000.pt \
        --metaXml_path=/data2/jxchen/jxchen_copy_1121/data/jxchen/dataset/dji/campus/metadata.xml \
        --ply_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/data/gt/yuehai_gt_label_fineReg.txt \
        --exp_name=logs_dji/test \
        --enable_semantic=True 




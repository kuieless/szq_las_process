export DETECTRON2_DATASETS=/data/yuqi/Datasets
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7

# 

output=/data/yuqi/code/GP-NeRF-semantic/logs_demo/0120_cuhksz_demo_geo_renderfar/mask2former_output

/data/yuqi/anaconda3/envs/mask2former/bin/python /data/yuqi/code/Mask2Former_a/demo/demo_zyq_0502_augment.py  \
    --config-file  /data/yuqi/code/Mask2Former_a/configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml \
    --input  /data/yuqi/code/GP-NeRF-semantic/logs_demo/0120_cuhksz_demo_geo_renderfar/pred_rgb  \
    --output  $output \
    --zyq_code  True  \
    --zyq_mapping True \
    --opts MODEL.WEIGHTS  /data/yuqi/code/Mask2Former_a/pretrained_ckpt/ade20k/model_final_e0c58e_panoptic_swinL.pkl

# cd $output
# ffmpeg -r 6 -pattern_type glob -i "*.jpg" -filter:v scale=-1:720  -vcodec libx264 -crf 18 -pix_fmt yuv420p ../alpha.mp4

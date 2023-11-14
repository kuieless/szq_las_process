####选择一个图像，往相机视角方向飞


import click
import os
import numpy as np
import cv2 as cv
from os.path import join as pjoin
from glob import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import torch
from gp_nerf.runner_gpnerf import Runner
from gp_nerf.opts import get_opts_base
from argparse import Namespace
from tqdm import tqdm
import cv2
from tools.unetformer.uavid2rgb import rgb2custom, custom2rgb,remapping

from PIL import Image
from pathlib import Path
import open3d as o3d
import pickle
import math
from dji.process_dji_v8_color import euler2rotation, rad
import xml.etree.ElementTree as ET
from collections import Counter
from pyntcloud import PyntCloud
import pandas as pd
from torch.nn.functional import interpolate
from mega_nerf.ray_utils import get_ray_directions
from tools.contrastive_lift.utils import cluster, visualize_panoptic_outputs
from torchvision.utils import make_grid
from gp_nerf.eval_utils import calculate_panoptic_quality_folders
from tools.contrastive_lift.utils import create_instances_from_semantics
from torch.utils.tensorboard import SummaryWriter
import configargparse


# torch.cuda.set_device(7)



def _get_train_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--num_points', type=int, default=200000,required=False, help='')
    parser.add_argument('--output_path', type=str, default='zyq/1113_campus_cross_view',required=False, help='')
    parser.add_argument('--panoptic_dir', type=str, default='logs_campus/1107_campus_density_depth_hash22_instance_origin_sam_0.001_depth_crossview/12/eval_100000_1112/panoptic',required=False, help='')
    
    parser.add_argument('--dataset_path', type=str, default='/data/yuqi/Datasets/DJI/Campus_new',required=False, help='')
    parser.add_argument('--all_centroids', type=str, default='',required=False, help='')
    
    parser.add_argument('--type', type=str, default='mean',required=False, choices=['mean', 'uniform'], help='')
    

        
    return parser.parse_args()



def hello(hparams) -> None:
    device='cpu'
    ### 只考虑building
    thing_classes = [1]

    bandwidth=0.2
    num_points=hparams.num_points
    use_dbscan=True
    use_silverman=False

    output_path = hparams.output_path
    panoptic_dir = hparams.panoptic_dir
    writer = SummaryWriter(os.path.join(output_path, 'tb'))

    if 'Yingrenshi' in hparams.dataset_path or 'Campus' in hparams.dataset_path:
        H, W = 912, 1368
    else:
        H, W = 1024, 1536


    sam_paths = []
    for ext in ('*.png', '*.npy'):
        sam_paths.extend(glob(os.path.join(hparams.dataset_path, 'val','instances_mask_0.001_depth', ext)))
    sam_paths.sort()

    sam_masks = []
    for idx, sam_path in enumerate(tqdm(sam_paths)):
        label = np.load(sam_path)
        label =  torch.tensor(label.astype(np.int32),dtype=torch.int32)
        sam_masks.append(label)
    eval_num = len(sam_masks)
    
    sam_masks = torch.stack(sam_masks).to(device).flatten()



    


    # cluster_sizes=np.arange(350, 2500, 100).tolist()
    cluster_sizes=[1000]

    for cluster_size in cluster_sizes:
        train_num = 0

        output=os.path.join(output_path, f'{hparams.type}_depth_{num_points}_{cluster_size}')

        # 这里创建文件夹用于存放聚类调试的结果
        Path(os.path.join(output)).mkdir(exist_ok=True)
        Path(os.path.join(output,'pred_semantics')).mkdir(exist_ok=True)
        Path(os.path.join(output,'pred_surrogateid')).mkdir(exist_ok=True)
        Path(os.path.join(output,'gt_semantics')).mkdir(exist_ok=True)
        Path(os.path.join(output,'gt_surrogateid')).mkdir(exist_ok=True)
        
        ### 先读入原始pred的rgb， semantic, instance, 以及对应的gt
        all_instance_features = np.load(os.path.join(panoptic_dir, "all_instance_features.npy"))
        all_points_semantics = np.load(os.path.join(panoptic_dir, "all_points_semantics.npy"))
        all_instance_features=torch.from_numpy(all_instance_features).to(device)
        
        all_points_rgb = np.load(os.path.join(panoptic_dir, "all_points_rgb.npy"))
        gt_points_rgb = np.load(os.path.join(panoptic_dir, "gt_points_rgb.npy"))
        gt_points_semantic = np.load(os.path.join(panoptic_dir, "gt_points_semantic.npy"))
        gt_points_instance = np.load(os.path.join(panoptic_dir, "gt_points_instance.npy"))

        all_points_semantics=torch.from_numpy(all_points_semantics).to(device).view(eval_num, H*W)
        all_points_rgb=torch.from_numpy(all_points_rgb).to(device).view(eval_num, H*W, -1)
        gt_points_semantic=torch.from_numpy(gt_points_semantic).to(device).view(eval_num, H*W)
        gt_points_rgb=torch.from_numpy(gt_points_rgb).to(device).view(eval_num, H*W, -1)
        gt_points_instance=torch.from_numpy(gt_points_instance).to(device).view(eval_num, H*W)


        ### 先对原始的pred instance feature， 用 pred semantic 过滤， 这里的 pred semantic 已经通过gt semantic过滤了
        all_thing_features = create_instances_from_semantics(all_instance_features, all_points_semantics.view(-1), thing_classes=thing_classes, device=device)
        
        #################################### 1 . mean 聚类
        with open(hparams.all_centroids, 'rb') as f:
            all_centroids = pickle.load(f)

        all_points_instances, centroids = cluster(all_thing_features.cpu().numpy(), bandwidth=bandwidth, device=device, num_images=eval_num+train_num, 
                                    num_points=num_points, use_silverman=use_silverman, use_dbscan=use_dbscan,cluster_size=cluster_size, all_centroids=all_centroids)
        

        save_i=0
        # for p_rgb, p_semantics, p_instances in zip(all_points_rgb, all_points_semantics, all_points_instances)
        for save_i in range(eval_num):
            p_rgb = all_points_rgb[save_i]
            p_semantics = all_points_semantics[save_i]

            p_instances = all_points_instances[save_i]

            gt_rgb = gt_points_rgb[save_i]
            gt_semantics = gt_points_semantic[save_i]
            gt_instances = gt_points_instance[save_i]

            output_semantics_with_invalid = p_semantics.detach()
            Image.fromarray(output_semantics_with_invalid.reshape(H, W).cpu().numpy().astype(np.uint8)).save(
                os.path.join(output , 'pred_semantics', ("%06d.png" % save_i)))
            
            Image.fromarray(p_instances.argmax(dim=1).reshape(H, W).cpu().numpy().astype(np.uint16)).save(
                os.path.join(output , 'pred_surrogateid', ("%06d.png" % save_i)))

            

            Image.fromarray(gt_semantics.reshape(H, W).cpu().numpy().astype(np.uint8)).save(
                os.path.join(output , 'gt_semantics', ("%06d.png" % save_i)))

            
            Image.fromarray(gt_instances.reshape(H, W).cpu().numpy().astype(np.uint16)).save(
                os.path.join(output , 'gt_surrogateid', ("%06d.png" % save_i)))        
        
        path_target_sem = os.path.join(output,'gt_semantics')
        path_target_inst = os.path.join(output,'gt_surrogateid')
        path_pred_sem = os.path.join(output,'pred_semantics')
        path_pred_inst = os.path.join(output,'pred_surrogateid')
        if Path(path_target_inst).exists():
            pq, sq, rq, metrics_each, pred_areas, target_areas, zyq_TP, zyq_FP, zyq_FN  = calculate_panoptic_quality_folders(path_pred_sem, path_pred_inst, 
                            path_target_sem, path_target_inst, image_size=[W,H])
            # val_metrics['pq'] = pq
            # val_metrics['sq'] = sq
            # val_metrics['rq'] = rq
            for key in metrics_each['all']:
                avg_val = metrics_each['all'][key]
                message = ' {}: {}'.format(key, avg_val)
                print(message)

            print(f'\n pq, sq, rq: {pq:.5f} {sq:.5f} {rq:.5f}\n')  
        
            with(Path(os.path.join(output, 'metric.txt'))).open('w') as f:
                f.write(f'\n pq, sq, rq: {pq:.5f} {sq:.5f} {rq:.5f}\n') 
                # f.write(f'panoptic metrics_each: {metrics_each} \n')  
                
                for key in metrics_each['all']:
                    avg_val = metrics_each['all'][key]
                    message = '      {}: {}'.format(key, avg_val)
                    f.write('{}\n'.format(message))
                    print(message)
                f.write('{}\n')
                f.write(f"pq, rq, sq, mIoU, TP, FP, FN: {metrics_each['all']['pq'][0].item()},{metrics_each['all']['rq'][0].item()},{metrics_each['all']['sq'][0].item()},{metrics_each['all']['iou_sum'][0].item()},{metrics_each['all']['true_positives'][0].item()},{metrics_each['all']['false_positives'][0].item()},{metrics_each['all']['false_negatives'][0].item()}\n")
                f.write('centroids_shape: {}\n'.format(centroids.shape))
                
            print('done')


            # 对TP进行处理
            TP = torch.tensor([value[1] for value in zyq_TP if value[0] == 1])
            FP = torch.tensor([value[1] for value in zyq_FP if value[0] == 1])
            FN = torch.tensor([value[1] for value in zyq_FN if value[0] == 1])

            for save_i in range(eval_num):
                p_rgb = all_points_rgb[save_i]
                p_semantics = all_points_semantics[save_i]
                p_instances = all_points_instances[save_i]
                gt_rgb = gt_points_rgb[save_i]
                gt_semantics = gt_points_semantic[save_i]
                gt_instances = gt_points_instance[save_i]
                stack = visualize_panoptic_outputs(
                    p_rgb.cpu(), p_semantics.cpu(), p_instances.cpu(), None, gt_rgb.cpu(), gt_semantics.cpu(), gt_instances.cpu(),
                    H, W, thing_classes=thing_classes, visualize_entropy=False,
                    TP=TP, FP=FP, FN=FN
                )
                grid = make_grid(stack, value_range=(0, 1), normalize=True, nrow=4).permute((1, 2, 0)).contiguous()
                grid = (grid * 255).cpu().numpy().astype(np.uint8)
                
                Image.fromarray(grid).save(os.path.join(output,("%06d.jpg" % save_i)))
            
            writer.add_scalar('pq/pq', metrics_each['all']['pq'][0].item(), cluster_size)
            writer.add_scalar('pq/sq', metrics_each['all']['sq'][0].item(), cluster_size)
            writer.add_scalar('pq/rq', metrics_each['all']['rq'][0].item(), cluster_size)

            writer.add_scalar('iou/iou_sum', metrics_each['all']['iou_sum'][0].item(), cluster_size)
            writer.add_scalar('iou/true_positives', metrics_each['all']['true_positives'][0], cluster_size)
            writer.add_scalar('iou/false_positives', metrics_each['all']['false_positives'][0],cluster_size)
            writer.add_scalar('iou/false_negatives', metrics_each['all']['false_negatives'][0],cluster_size)

            writer.add_scalar('centroids', centroids.shape[0], cluster_size)

                
        with open(os.path.join(output, 'test_centroids.npy'), "wb") as file:
            pickle.dump(centroids, file)


if __name__ == '__main__':

    hello(_get_train_opts())

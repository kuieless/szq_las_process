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

from torch.utils.tensorboard import SummaryWriter

# torch.cuda.set_device(7)



def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--num_points', type=int, default=50000,required=False, help='')
    parser.add_argument('--cluster_size', type=int, default=500,required=False, help='experiment name')
    
    parser.add_argument('--all_thing_features_15', type=str, default='logs_dji/1026_yingrenshi_density_depth_hash22_instance_freeze_gt_slow_shuffle/0/eval_80000/panoptic/all_thing_features.npy',required=False, help='experiment name')
    parser.add_argument('--sem_rgb_15', type=str, default='logs_dji/1026_yingrenshi_density_depth_hash22_instance_freeze_gt_slow_shuffle/0/eval_80000/panoptic',required=False, help='experiment name')
    parser.add_argument('--all_thing_features_fushi', type=str, default='logs_dji/1026_yingrenshi_density_depth_hash22_instance_freeze_gt_slow_shuffle/2/eval_80000/panoptic/all_thing_features.npy',required=False, help='experiment name')
    parser.add_argument('--sem_rgb_fushi', type=str, default='logs_dji/1026_yingrenshi_density_depth_hash22_instance_freeze_gt_slow_shuffle/2/eval_80000/panoptic',required=False, help='experiment name')
    
    # parser.add_argument('--sem_rgb_15', type=str, default='logs_dji/1025_yingrenshi_density_depth_hash22_instance_freeze_gt_slow/0/eval_170000/panoptic',required=False, help='experiment name')
    # parser.add_argument('--all_thing_features_15', type=str, default='logs_dji/1025_yingrenshi_density_depth_hash22_instance_freeze_gt_slow/0/eval_170000/panoptic/all_thing_features.npy',required=False, help='experiment name')
    
    # parser.add_argument('--sem_rgb_fushi', type=str, default='logs_dji/1025_yingrenshi_density_depth_hash22_instance_freeze_gt_slow/1/eval_170000/panoptic',required=False, help='experiment name')
    # parser.add_argument('--all_thing_features_fushi', type=str, default='logs_dji/1025_yingrenshi_density_depth_hash22_instance_freeze_gt_slow/1/eval_170000/panoptic/all_thing_features.npy',required=False, help='experiment name')
        
    return parser.parse_args()


def hello(hparams: Namespace) -> None:
    num_points=hparams.num_points
    cluster_size=hparams.cluster_size



    writer = SummaryWriter(os.path.join('zyq_cluster', 'tb'))
    

    # for cluster_size in [500,1000,2000,3000,4000,5000,7000,10000]:
    for cluster_size in [500,1000,2000]:
        device='cuda'
        bandwidth=0.2
        use_dbscan=True
        use_silverman=False

        output=f'1025_panoptic_{num_points}point_clustersize{cluster_size}'

        Path(os.path.join('zyq_cluster',output)).mkdir(exist_ok=True, parents=True)
        Path(os.path.join('zyq_cluster',output,'pred_semantics')).mkdir(exist_ok=True)
        Path(os.path.join('zyq_cluster',output,'pred_surrogateid')).mkdir(exist_ok=True)
        Path(os.path.join('zyq_cluster',output,'gt_semantics')).mkdir(exist_ok=True)
        Path(os.path.join('zyq_cluster',output,'gt_surrogateid')).mkdir(exist_ok=True)

        #  test图像计算的instance feature，用来聚类
        all_thing_features = np.load(hparams.all_thing_features_15)

        H, W = 912, 1368
        thing_classes = [1]

        all_points_instances_15, centroids = cluster(all_thing_features, bandwidth=bandwidth, device=device, num_images=15, 
                                    num_points=num_points, use_silverman=use_silverman, use_dbscan=use_dbscan,cluster_size=cluster_size)

        # 得到聚类后， 读入俯视图的数据
        all_thing_features = np.load(hparams.all_thing_features_fushi)

        all_points_instances, _ = cluster(all_thing_features, bandwidth=bandwidth, device=device, num_images=1, 
                                    num_points=num_points, use_silverman=use_silverman, use_dbscan=use_dbscan,cluster_size=cluster_size, all_centroids=centroids)



        output_dir = hparams.sem_rgb_fushi
        all_points_semantics = np.load(os.path.join(output_dir, "all_points_semantics.npy"))
        all_points_rgb = np.load(os.path.join(output_dir, "all_points_rgb.npy"))
        all_points_semantics=torch.from_numpy(all_points_semantics).to(device).view(1, 912*1368)
        all_points_rgb=torch.from_numpy(all_points_rgb).to(device).view(1, 912*1368, -1)

        for save_i in range(1):
            p_rgb = all_points_rgb[save_i]
            p_semantics = all_points_semantics[save_i]
            p_instances = all_points_instances[save_i]


            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            import matplotlib.pyplot as plt
            lda = LinearDiscriminantAnalysis(n_components=2)
            lda_result = lda.fit_transform(all_thing_features[:,1:], p_instances.argmax(dim=1).cpu().numpy())
            

            plt.scatter(lda_result[:, 0], lda_result[:, 1], c=p_instances.argmax(dim=1).cpu().numpy(), cmap='viridis')
            plt.xlabel('LDA Component 1')
            plt.ylabel('LDA Component 2')
            plt.title('LDA Visualization')

            # 保存图像为文件（例如，PNG格式）
            plt.savefig(os.path.join('zyq_cluster', output,f'lda_visualization_{num_points}point_clustersize{cluster_size}.png'))

            # 关闭图形窗口（如果不需要显示图形界面）
            plt.close()

            output_semantics_with_invalid = p_semantics.detach()
            Image.fromarray(output_semantics_with_invalid.reshape(H, W).cpu().numpy().astype(np.uint8)).save(
                os.path.join('zyq_cluster', output , 'pred_semantics', ("%06d.png" % save_i)))
            
            Image.fromarray(p_instances.argmax(dim=1).reshape(H, W).cpu().numpy().astype(np.uint16)).save(
                os.path.join('zyq_cluster', output , 'pred_surrogateid', ("%06d.png" % save_i)))

            stack = visualize_panoptic_outputs(
                p_rgb, p_semantics, p_instances, None, None,None,None,
                H, W, thing_classes=thing_classes, visualize_entropy=False
            )
            grid = make_grid(stack, value_range=(0, 1), normalize=True, nrow=3).permute((1, 2, 0)).contiguous()
            grid = (grid * 255).cpu().numpy().astype(np.uint8)
            
            Image.fromarray(grid).save(os.path.join('zyq_cluster',output,("%06d_fushitu.jpg" % save_i)))

        output_dir = hparams.sem_rgb_15
        all_thing_features = np.load(os.path.join(output_dir, "all_thing_features.npy"))
        all_points_semantics = np.load(os.path.join(output_dir, "all_points_semantics.npy"))
        all_points_rgb = np.load(os.path.join(output_dir, "all_points_rgb.npy"))
        all_points_semantics=torch.from_numpy(all_points_semantics).to(device).view(15, 912*1368)
        all_points_rgb=torch.from_numpy(all_points_rgb).to(device).view(15, 912*1368, -1)
        gt_points_rgb = np.load(os.path.join(output_dir, "gt_points_rgb.npy"))
        gt_points_semantic = np.load(os.path.join(output_dir, "gt_points_semantic.npy"))
        gt_points_instance = np.load(os.path.join(output_dir, "gt_points_instance.npy"))
        gt_points_semantic=torch.from_numpy(gt_points_semantic).to(device).view(15, 912*1368)
        gt_points_rgb=torch.from_numpy(gt_points_rgb).to(device).view(15, 912*1368, -1)
        gt_points_instance=torch.from_numpy(gt_points_instance).to(device).view(15, 912*1368)

        save_i=0
        # for p_rgb, p_semantics, p_instances in zip(all_points_rgb, all_points_semantics, all_points_instances)
        for save_i in range(15):
            p_rgb = all_points_rgb[save_i]
            # p_semantics = all_points_semantics[save_i]
            p_semantics = gt_points_semantic[save_i]

            p_instances = all_points_instances_15[save_i]

            gt_rgb = gt_points_rgb[save_i]
            gt_semantics = gt_points_semantic[save_i]
            gt_instances = gt_points_instance[save_i]

            output_semantics_with_invalid = p_semantics.detach()
            Image.fromarray(output_semantics_with_invalid.reshape(H, W).cpu().numpy().astype(np.uint8)).save(
                os.path.join('zyq_cluster', output , 'pred_semantics', ("%06d.png" % save_i)))
            
            Image.fromarray(p_instances.argmax(dim=1).reshape(H, W).cpu().numpy().astype(np.uint16)).save(
                os.path.join('zyq_cluster', output , 'pred_surrogateid', ("%06d.png" % save_i)))

            

            Image.fromarray(gt_semantics.reshape(H, W).cpu().numpy().astype(np.uint8)).save(
                os.path.join('zyq_cluster', output , 'gt_semantics', ("%06d.png" % save_i)))

            
            Image.fromarray(gt_instances.reshape(H, W).cpu().numpy().astype(np.uint16)).save(
                os.path.join('zyq_cluster', output , 'gt_surrogateid', ("%06d.png" % save_i)))        
            
            stack = visualize_panoptic_outputs(
                p_rgb, p_semantics, p_instances, None, gt_rgb, gt_semantics, gt_instances,
                H, W, thing_classes=thing_classes, visualize_entropy=False
            )
            grid = make_grid(stack, value_range=(0, 1), normalize=True, nrow=3).permute((1, 2, 0)).contiguous()
            grid = (grid * 255).cpu().numpy().astype(np.uint8)
            
            Image.fromarray(grid).save(os.path.join('zyq_cluster',output,("%06d.jpg" % save_i)))


        path_target_sem = os.path.join('zyq_cluster',output,'gt_semantics')
        path_target_inst = os.path.join('zyq_cluster',output,'gt_surrogateid')
        path_pred_sem = os.path.join('zyq_cluster',output,'pred_semantics')
        path_pred_inst = os.path.join('zyq_cluster',output,'pred_surrogateid')
        if Path(path_target_inst).exists():
            pq, sq, rq, metrics_each = calculate_panoptic_quality_folders(path_pred_sem, path_pred_inst, 
                            path_target_sem, path_target_inst, image_size=[W,H])
            with(Path(os.path.join('zyq_cluster',output, 'metric.txt'))).open('w') as f:
                for key in metrics_each['all']:
                    avg_val = metrics_each['all'][key]
                    message = ' {}: {}'.format(key, avg_val)
                    f.write('{}\n'.format(message))
                    print(message)


                print(f'\n pq, sq, rq: {pq:.5f} {sq:.5f} {rq:.5f}\n')  
                f.write(f'\n pq, sq, rq: {pq:.5f} {sq:.5f} {rq:.5f}\n')  
                print('done')
                f.write(f"centroids shape: {centroids.shape}")
                print(f"centroids shape: {centroids.shape}")
                
                writer.add_scalar('pq/pq', pq, cluster_size)
                writer.add_scalar('pq/sq', sq, cluster_size)
                writer.add_scalar('pq/rq', rq, cluster_size)

                writer.add_scalar('iou/iou_sum', metrics_each['all']['iou_sum'][0], cluster_size)
                writer.add_scalar('iou/true_positives', metrics_each['all']['true_positives'][0], cluster_size)
                writer.add_scalar('iou/false_positives', metrics_each['all']['false_positives'][0],cluster_size)
                writer.add_scalar('iou/false_negatives', metrics_each['all']['false_negatives'][0],cluster_size)

                writer.add_scalar('centroids', centroids.shape[0],cluster_size)



if __name__ == '__main__':

    hello(_get_train_opts())

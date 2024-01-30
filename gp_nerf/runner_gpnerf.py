import datetime
import faulthandler
import math
import os
import random
import shutil
import signal
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm
import gc

from gp_nerf.datasets.filesystem_dataset import FilesystemDataset

from gp_nerf.image_metadata import ImageMetadata
from mega_nerf.metrics import psnr, ssim, lpips
from mega_nerf.misc_utils import main_print, main_tqdm
from mega_nerf.ray_utils import get_rays, get_ray_directions
from gp_nerf.models.model_utils import get_nerf, get_bg_nerf

import wandb
from torchvision.utils import make_grid

#semantic
from tools.unetformer.uavid2rgb import uavid2rgb, custom2rgb, remapping, remapping_remove_ground
from tools.unetformer.uavid2rgb import custom2rgb_point
from tools.unetformer.metric import Evaluator

import pandas as pd

from gp_nerf.eval_utils import get_depth_vis, get_semantic_gt_pred, get_sdf_normal_map, get_semantic_gt_pred_render_zyq, eval_others, get_instance_pred, calculate_panoptic_quality_folders
from tools.contrastive_lift.utils import cluster, visualize_panoptic_outputs, assign_clusters
from gp_nerf.eval_utils import calculate_metric_rendering, write_metric_to_folder_logger, save_semantic_metric
from gp_nerf.eval_utils import prepare_depth_normal_visual


from gp_nerf.sa3d_utils import seg_loss, _generate_index_matrix, prompting_coarse_N, prompting_coarse
from tools.segment_anything import sam_model_registry, SamPredictor

# nr3d_sdf
from typing import Literal, Union
from nr3d_lib.logger import Logger
from nr3d_lib.models.loss.safe import safe_mse_loss
from gp_nerf.sample_bg import contract_to_unisphere_new
import trimesh
from scripts.extract_mesh_sdf import extract_geometry
from scripts.visualize_points import visualize_points
import open3d as o3d
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
import pickle

from tools.contrastive_lift.utils import create_instances_from_semantics

import xml.etree.ElementTree as ET
from pyntcloud import PyntCloud


to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            # print(s)
            nn = nn*s
        pp += nn
    return pp
    

def custom_collate(batch):
    # 过滤掉为 None 的项
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    return torch.utils.data.dataloader.default_collate(batch)


def init_predictor(device):
    sam_checkpoint = "tools/segment_anything/sam_vit_h_4b8939.pth"
    
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print("SAM initializd.")
    predictor = SamPredictor(sam)
    return predictor



def contrastive_loss(features, instance_labels, temperature):
    bsize = features.size(0)
    masks = instance_labels.view(-1, 1).repeat(1, bsize).eq_(instance_labels.clone())
    masks = masks.fill_diagonal_(0, wrap=False)

    # compute similarity matrix based on Euclidean distance
    distance_sq = torch.pow(features.unsqueeze(1) - features.unsqueeze(0), 2).sum(dim=-1)
    # temperature = 1 for positive pairs and temperature for negative pairs
    temperature = torch.ones_like(distance_sq) * temperature
    temperature = torch.where(masks==1, temperature, torch.ones_like(temperature))

    similarity_kernel = torch.exp(-distance_sq/temperature)
    logits = torch.exp(similarity_kernel)

    p = torch.mul(logits, masks).sum(dim=-1)
    Z = logits.sum(dim=-1)

    prob = torch.div(p, Z)
    prob_masked = torch.masked_select(prob, prob.ne(0))
    loss = -prob_masked.log().sum()/bsize
    return loss


def rad(x):
    return math.radians(x)

def zhitu_2_nerf(dataset_path, metaXml_path, points_xyz):
    # to process points_rgb
    coordinate_info = torch.load(dataset_path + '/coordinates.pt')
    origin_drb = coordinate_info['origin_drb'].numpy()
    pose_scale_factor = coordinate_info['pose_scale_factor']

    root = ET.parse(metaXml_path).getroot()
    translation = np.array(root.find('SRSOrigin').text.split(',')).astype(np.float)    

    #######################################
    ZYQ = torch.DoubleTensor([[0, 0, -1],
                            [0, 1, 0],
                            [1, 0, 0]])
    ZYQ_1 = torch.DoubleTensor([[1, 0, 0],
                            [0, math.cos(rad(135)), math.sin(rad(135))],
                            [0, -math.sin(rad(135)), math.cos(rad(135))]])      
    # points_nerf = np.array(xyz)
    points_nerf = points_xyz
    points_nerf += translation
    points_nerf = ZYQ.numpy() @ points_nerf.T
    points_nerf = (ZYQ_1.numpy() @ points_nerf).T
    points_nerf = (points_nerf - origin_drb) / pose_scale_factor
    return points_nerf


class Runner:
    def __init__(self, hparams: Namespace, set_experiment_path: bool = True):
        faulthandler.register(signal.SIGUSR1)
        print(f"ignore_index: {hparams.ignore_index}")
        self.temperature = 100
        if hparams.only_train_building:
            self.thing_classes=[1]
        else:
            self.thing_classes=[0,1,2,3,4]
        # use when instance_loss_mode == 'linear_assignment'
        self.loss_instances_cluster = torch.nn.CrossEntropyLoss(reduction='none')

        if hparams.balance_weight:
            # 1017之前：cluster 1，  building 1， road 1， car 5， tree 5， vegetation 5   
            # balance_weight = torch.FloatTensor([1, 1, 1, 5, 5, 5, 1, 1, 1, 1, 1]).cuda()
            # 1017测试增大car的比例：cluster 1，  building 1， road 1， car 2， tree 1， vegetation 1
            balance_weight = torch.FloatTensor([1, 1, 1, 2, 1]).cuda()
            # balance_weight = torch.FloatTensor([1, 1, 2, 2, 1]).cuda()

            CrossEntropyLoss = nn.CrossEntropyLoss(weight=balance_weight, ignore_index=hparams.ignore_index)
        else:
            CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=hparams.ignore_index)

        self.crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label)
        self.logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
        
        self.group_loss_feat = lambda pred, target: nn.CosineSimilarity(dim=1)(pred, target)

        self.color_list = torch.randint(0, 255,(100, 3)).to(torch.float32)

        if hparams.depth_loss:
            from gp_nerf.loss_monosdf import ScaleAndShiftInvariantLoss
            self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)

        if hparams.normal_loss:
            from gp_nerf.loss_monosdf import get_l1_normal_loss
            self.normal_loss = get_l1_normal_loss

        if hparams.ckpt_path is not None:
            checkpoint = torch.load(hparams.ckpt_path, map_location='cpu')
            np.random.set_state(checkpoint['np_random_state'])     
            torch.set_rng_state(checkpoint['torch_random_state'])  
            random.setstate(checkpoint['random_state'])
        else:
            np.random.seed(hparams.random_seed)
            torch.manual_seed(hparams.random_seed)
            random.seed(hparams.random_seed)

        self.hparams = hparams
        if self.hparams.dataset_type == 'sam' and self.hparams.add_random_rays:
            assert hparams.sample_random_num % hparams.batch_size ==0
            self.hparams.sample_random_num_each = int(hparams.sample_random_num / hparams.batch_size)
 
        

        # self.hparams.train_iterations = int(self.hparams.train_iterations / (self.hparams.batch_size/1024))
        # self.hparams.val_interval = self.hparams.train_iterations + 1

        if 'RANK' in os.environ:
            dist.init_process_group(backend='nccl', timeout=datetime.timedelta(0, hours=24))
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            self.is_master = (int(os.environ['RANK']) == 0)
        else:
            self.is_master = True

        self.is_local_master = ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0
        main_print(hparams)

        if set_experiment_path:
            self.experiment_path = self._get_experiment_path() if self.is_master else None
            self.model_path = self.experiment_path / 'models' if self.is_master else None

        
        self.wandb = None
        self.writer = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if 'llff' in self.hparams.dataset_type:
            
            from gp_nerf.datasets.llff import NeRFDataset
            dataset = NeRFDataset(self.hparams, device=self.device, type='all')
            
            self.sphere_center = None
            self.sphere_radius = None
            self.nerf = get_nerf(hparams, len(dataset)).to(self.device)
            self.bg_nerf = None
            
        else:
            coordinate_info = torch.load(Path(hparams.dataset_path) / 'coordinates.pt', map_location='cpu')
            self.origin_drb = coordinate_info['origin_drb']
            self.pose_scale_factor = coordinate_info['pose_scale_factor']
            main_print('Origin: {}, scale factor: {}'.format(self.origin_drb, self.pose_scale_factor))

            self.near = (hparams.near / self.pose_scale_factor)

            if self.hparams.far is not None:
                self.far = hparams.far / self.pose_scale_factor
            elif hparams.bg_nerf:
                self.far = 1e5
            elif hparams.gpnerf:
                self.far = 1e5
            else:
                self.far = 2
            main_print('Ray bounds: {}, {}'.format(self.near, self.far))

            # mesh = trimesh.load_mesh('/data/jxchen/GPNeRF-semantic/residence/all/residence_meshlabnorm.obj')
            # pts_3d = mesh.vertices
            # pts_3d = np.array(pts_3d) * self.pose_scale_factor + self.origin_drb[0].numpy()
            # np.savetxt('tran_meshpoint.txt', pts_3d)

             
            self.ray_altitude_range = [(x - self.origin_drb[0]) / self.pose_scale_factor for x in
                                    hparams.ray_altitude_range] if hparams.ray_altitude_range is not None else None
            main_print('Ray altitude range in [-1, 1] space: {}'.format(self.ray_altitude_range))
            main_print('Ray altitude range in metric space: {}'.format(hparams.ray_altitude_range))

            if self.ray_altitude_range is not None:
                assert self.ray_altitude_range[0] < self.ray_altitude_range[1]


            if self.hparams.cluster_mask_path is not None:
                cluster_params = torch.load(Path(self.hparams.cluster_mask_path).parent / 'params.pt', map_location='cpu')
                assert cluster_params['near'] == self.near
                assert (torch.allclose(cluster_params['origin_drb'], self.origin_drb))
                assert cluster_params['pose_scale_factor'] == self.pose_scale_factor

                if self.ray_altitude_range is not None:
                    assert (torch.allclose(torch.FloatTensor(cluster_params['ray_altitude_range']),
                                        torch.FloatTensor(self.ray_altitude_range))), \
                        '{} {}'.format(self.ray_altitude_range, cluster_params['ray_altitude_range'])

            self.train_items, self.val_items = self._get_image_metadata()



            main_print('Using {} train images and {} val images'.format(len(self.train_items), len(self.val_items)))

            camera_positions = torch.cat([x.c2w[:3, 3].unsqueeze(0) for x in self.train_items + self.val_items])
            min_position = camera_positions.min(dim=0)[0]
            max_position = camera_positions.max(dim=0)[0]

            main_print('Camera range in metric space: {} {}'.format(min_position * self.pose_scale_factor + self.origin_drb,
                                                                    max_position * self.pose_scale_factor + self.origin_drb))

            main_print('Camera range in [-1, 1] space: {} {}'.format(min_position, max_position))

            if hparams.ellipse_bounds or (hparams.ellipse_bounds and hparams.gpnerf):
                assert hparams.ray_altitude_range is not None

                if self.ray_altitude_range is not None:
                    ground_poses = camera_positions.clone()
                    ground_poses[:, 0] = self.ray_altitude_range[1]
                    air_poses = camera_positions.clone()
                    air_poses[:, 0] = self.ray_altitude_range[0]
                    used_positions = torch.cat([camera_positions, air_poses, ground_poses])
                else:
                    used_positions = camera_positions

                max_position[0] = self.ray_altitude_range[1]
                main_print('Camera range in [-1, 1] space with ray altitude range: {} {}'.format(min_position,
                                                                                                max_position))

                self.sphere_center = ((max_position + min_position) * 0.5).to(self.device)
                self.sphere_radius = ((max_position - min_position) * 0.5).to(self.device)
                scale_factor = ((used_positions.to(self.device) - self.sphere_center) / self.sphere_radius).norm(
                    dim=-1).max()
                self.sphere_radius *= (scale_factor * hparams.ellipse_scale_factor)
                main_print('Sphere center: {}, radius: {}'.format(self.sphere_center, self.sphere_radius))
                hparams.z_range = self.ray_altitude_range
                hparams.sphere_center=self.sphere_center
                hparams.sphere_radius=self.sphere_radius
                hparams.aabb_bound = max(self.sphere_radius)
                # fg_box_bound
                hparams.fg_box_bound = np.array([[0.0094, -0.5091, -1.0803], [0.2063, 0.5585, 1.1147]])
                # hparams.fg_box_bound[0,:] = hparams.fg_box_bound[0,:] - abs(hparams.fg_box_bound[0,:]*0.1)
                # hparams.fg_box_bound[1,:] = hparams.fg_box_bound[1,:] + abs(hparams.fg_box_bound[1,:]*0.1)

                #aabb for nr3d 
                z_range = torch.tensor(hparams.z_range, dtype=torch.float32)
                hparams.stretch = torch.tensor([[z_range[0], hparams.sphere_center[1] - hparams.sphere_radius[1], hparams.sphere_center[2] - hparams.sphere_radius[2]], 
                                               [z_range[1], hparams.sphere_center[1] + hparams.sphere_radius[1], hparams.sphere_center[2] + hparams.sphere_radius[2]]]).to(self.device)
                hparams.pose_scale_factor = self.pose_scale_factor

            else:
                self.sphere_center = None
                self.sphere_radius = None
            # if self.hparams.dataset_type == 'sam':
            #     self.nerf = get_nerf(hparams, len(self.train_items+self.val_items)).to(self.device)
            # else:
            self.nerf = get_nerf(hparams, len(self.train_items)).to(self.device)

            if 'RANK' in os.environ:
                self.nerf = torch.nn.parallel.DistributedDataParallel(self.nerf, device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                    output_device=int(os.environ['LOCAL_RANK']))

            if hparams.bg_nerf:
                self.bg_nerf = get_bg_nerf(hparams, len(self.train_items)).to(self.device)
                if 'RANK' in os.environ:
                    self.bg_nerf = torch.nn.parallel.DistributedDataParallel(self.bg_nerf,
                                                                            device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                            output_device=int(os.environ['LOCAL_RANK']))
            else:
                self.bg_nerf = None


            if hparams.bg_nerf:
                bg_parameters = get_n_params(self.bg_nerf)
            else:
                bg_parameters = 0
            fg_parameters = get_n_params(self.nerf)
            print("the parameters of whole model:\t total: {}, fg: {}, bg: {}".format(fg_parameters+bg_parameters,fg_parameters,bg_parameters))
            if self.wandb is not None:
                self.wandb.log({"parameters/fg": fg_parameters})
                self.wandb.log({"parameters/bg": bg_parameters})


    def query_nerf_result(self):
        with torch.no_grad():
            nerf = self.nerf
            hparams = self.hparams
            output_path = hparams.output_path
            metaXml_path = hparams.metaXml_path
            ply_path = hparams.ply_path
            

            # if not os.path.exists(output_path):
            #     os.makedirs(output_path)


            # pointcloud
            ply_data = np.loadtxt(ply_path)
            points_xyz_raw = ply_data[:,:3]
            points_xyz = zhitu_2_nerf(hparams.dataset_path, metaXml_path, points_xyz_raw)
            xyz_ = torch.from_numpy(points_xyz).to(self.device)
            xyz_ = contract_to_unisphere_new(xyz_,hparams)
            out_chunks = []
            out_semantic_chunk = [] 
            B = xyz_.shape[0]
            model_chunk_size = 1024*1024
            for i in range(0, B, model_chunk_size):

                xyz_chunk = xyz_[i:i + model_chunk_size]
                chunk_size = xyz_chunk.shape[0]
                rays_d_ = torch.zeros((chunk_size,3)).to(self.device)
                image_indices_ = torch.ones((chunk_size,1)).to(self.device)

                xyz_chunk = torch.cat([xyz_chunk,
                                    rays_d_,
                                    image_indices_], 1)
                
                sigma_noise=None

                    
                model_chunk, semantic_chunk= nerf('fg', xyz_chunk.float(), sigma_noise=sigma_noise, train_iterations=200000)
                out_chunks += [model_chunk]
                out_semantic_chunk += [semantic_chunk]
            
            out_semantic = torch.cat(out_semantic_chunk, 0)
            

                # alphas = 0
                # T = torch.cumprod(1 - alphas + 1e-8, -1)
                # T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
                # weights = alphas * T
                # sem_map = torch.sum(weights[..., None] * sem_logits, -2)
                
            # ?
            sem_label = self.logits_2_label(out_semantic)
            sem_label = remapping(sem_label)

            sem_rgb = custom2rgb_point(sem_label.cpu().numpy())

            print('save pc...')
            cloud = PyntCloud(pd.DataFrame(
                # same arguments that you are passing to visualize_pcl
                data=np.hstack((ply_data[:,:3], np.uint8(sem_rgb))),
                columns=["x", "y", "z", "red", "green", "blue"]))
            cloud.to_file(f"{hparams.output_path}")

            print('done')



    def train(self):

        self._setup_experiment_dir()
        scaler = torch.cuda.amp.GradScaler(enabled=self.hparams.amp)

        if (self.hparams.enable_semantic or self.hparams.enable_instance) and self.hparams.freeze_geo and self.hparams.ckpt_path is not None:
            
            for p_base in self.nerf.encoder_bg.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.plane_encoder.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.sigma_net.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.sigma_net_bg.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.color_net.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.color_net_bg.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.embedding_a.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.embedding_xyz.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.encoder_dir.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.encoder_dir_bg.parameters():
                p_base.requires_grad = False
            if 'nr3d' in self.hparams.network_type:
                for p_base in self.nerf.encoding.parameters():
                    p_base.requires_grad = False
                for p_base in self.nerf.decoder.parameters():
                    p_base.requires_grad = False
            else:
                for p_base in self.nerf.encoder.parameters():
                    p_base.requires_grad = False
            # 训练instance，冻结semantic
            if self.hparams.freeze_semantic and 'InstanceBuilding' not in self.hparams.dataset_path:
                for p_base in self.nerf.semantic_linear.parameters():
                    p_base.requires_grad = False
                for p_base in self.nerf.semantic_linear_bg.parameters():
                    p_base.requires_grad = False
                
                
            non_frozen_parameters = [p for p in self.nerf.parameters() if p.requires_grad]
            optimizers = {}
            # optimizers['nerf'] = Adam(non_frozen_parameters, lr=self.hparams.lr)
            optimizers['nerf'] = torch.optim.SGD(non_frozen_parameters, lr=self.hparams.lr)
            
        else:
            optimizers = {}
            optimizers['nerf'] = Adam(self.nerf.parameters(), lr=self.hparams.lr)
        
        if self.bg_nerf is not None:
            optimizers['bg_nerf'] = Adam(self.bg_nerf.parameters(), lr=self.hparams.lr)

        if self.hparams.ckpt_path is not None:
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            # # add by zyq : load the pretrain-gpnerf to train the semantic
            # if self.hparams.resume_ckpt_state:
            # if not (self.hparams.enable_semantic or self.hparams.enable_instance):
            if not (self.hparams.enable_semantic or self.hparams.enable_instance) or self.hparams.continue_train:
                train_iterations = checkpoint['iteration']
                for key, optimizer in optimizers.items():
                    optimizer_dict = optimizer.state_dict()
                    optimizer_dict.update(checkpoint['optimizers'][key])
                    optimizer.load_state_dict(optimizer_dict)
            else:
                print(f'load weights from {self.hparams.ckpt_path}, strat training from 0')
                train_iterations = 0


            scaler_dict = scaler.state_dict()
            scaler_dict.update(checkpoint['scaler'])
            scaler.load_state_dict(scaler_dict)
            
            discard_index = checkpoint['dataset_index'] if (self.hparams.resume_ckpt_state and not self.hparams.debug) and (not self.hparams.freeze_geo) else -1
            print(f"dicard_index:{discard_index}")
        else:
            train_iterations = 0
            discard_index = -1

        schedulers = {}
        for key, optimizer in optimizers.items():
            # schedulers[key] = ExponentialLR(optimizer,
            #                                 gamma=self.hparams.lr_decay_factor ** (1 / self.hparams.train_iterations),
            #                                 last_epoch=train_iterations - 1)
            schedulers[key] = StepLR(optimizer,
                                     step_size=self.hparams.train_iterations / 4,
                                     gamma=self.hparams.lr_decay_factor,
                                     last_epoch=train_iterations - 1)
            

        # load data
        if self.hparams.dataset_type == 'filesystem':
            # Let the local master write data to disk first
            # We could further parallelize the disk writing process by having all of the ranks write data,
            # but it would make determinism trickier
            if 'RANK' in os.environ and (not self.is_local_master):
                dist.barrier()

            dataset = FilesystemDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                        self.hparams.center_pixels, self.device,
                                        [Path(x) for x in sorted(self.hparams.chunk_paths)], self.hparams.num_chunks,
                                        self.hparams.train_scale_factor, self.hparams.disk_flush_size,self.hparams.desired_chunks, self.hparams)
            if self.hparams.ckpt_path is not None and ((self.hparams.resume_ckpt_state and not self.hparams.debug) and (not self.hparams.freeze_geo)):
                dataset.set_state(checkpoint['dataset_state'])
            if 'RANK' in os.environ and self.is_local_master:
                dist.barrier()
        elif self.hparams.dataset_type == 'memory':
            from gp_nerf.datasets.memory_dataset import MemoryDataset
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
        elif self.hparams.dataset_type == 'memory_depth_dji':
            from gp_nerf.datasets.memory_dataset_depth_dji import MemoryDataset
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
        elif self.hparams.dataset_type == 'memory_depth_dji_instance':
            if 'InstanceBuilding' not in self.hparams.dataset_path:
                from gp_nerf.datasets.memory_dataset_depth_dji_instance import MemoryDataset
            else:
                from gp_nerf.datasets.memory_dataset_depth_dji_instance_IB import MemoryDataset
                
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
            self.H = dataset.H
            self.W = dataset.W
        elif self.hparams.dataset_type == 'memory_depth_dji_instance_crossview':
            if 'InstanceBuilding' not in self.hparams.dataset_path:
                from gp_nerf.datasets.memory_dataset_depth_dji_instance_crossview import MemoryDataset
            else:
                from gp_nerf.datasets.memory_dataset_depth_dji_instance_crossview_IB import MemoryDataset
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
            self.H = dataset.H
            self.W = dataset.W
        elif self.hparams.dataset_type == 'memory_depth_dji_instance_crossview_process':
            from gp_nerf.datasets.memory_dataset_depth_dji_instance_crossview_process import MemoryDataset
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
            self.H = dataset.H
            self.W = dataset.W
        else:
            raise Exception('Unrecognized dataset type: {}'.format(self.hparams.dataset_type))

        if self.is_master:
            pbar = tqdm(total=self.hparams.train_iterations)
            pbar.update(train_iterations)
        else:
            pbar = None
        # start training

        chunk_id = 0
        while train_iterations < self.hparams.train_iterations:

            # If discard_index >= 0, we already set to the right chunk through set_state
            if self.hparams.dataset_type == 'filesystem' and discard_index == -1:
                dataset.load_chunk()

            if 'RANK' in os.environ:
                world_size = int(os.environ['WORLD_SIZE'])
                sampler = DistributedSampler(dataset, world_size, int(os.environ['RANK']))
                assert self.hparams.batch_size % world_size == 0
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size // world_size, sampler=sampler,
                                         num_workers=0, pin_memory=True)
            else:
                if 'sam' in self.hparams.dataset_type:
                    data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0,
                                                pin_memory=False, collate_fn=custom_collate)
                elif self.hparams.dataset_type == 'memory_depth':
                    data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0,
                                                pin_memory=False)
                elif 'llff' in self.hparams.dataset_type or self.hparams.dataset_type =='mega_sa3d':
                    data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0,
                                                pin_memory=False, collate_fn=custom_collate)
                elif self.hparams.dataset_type == 'memory_depth_dji_instance':
                    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                                pin_memory=False, collate_fn=custom_collate)
                elif self.hparams.dataset_type =='memory_depth_dji_instance_crossview':
                    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                                                pin_memory=False, collate_fn=custom_collate)
                elif self.hparams.dataset_type == 'memory_depth_dji_instance_crossview_process':
                    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                                pin_memory=False, collate_fn=custom_collate)
                    
                else:
                    data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=16,
                                                pin_memory=False)

            if train_iterations == 0 and self.hparams.network_type == 'sdf_nr3d':
                if self.hparams.geo_init_method == 'road_surface':
                    self.pretrain_sdf_road_surface_2d(floor_up_sign=-1, ego_height=0)
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        self.visualize_sdf_surface(self.nerf, save_path='./output/sdf.obj', resolution=256, threshold=0, device=self.device)
            for dataset_index, item in enumerate(data_loader): #, start=10462):
                # if dataset_index < 48000:
                #     train_iterations += 1
                #     pbar.update(1)
                #     continue
                # torch.cuda.empty_cache()
                # if self.hparams.dataset_type == 'memory_depth_dji_instance_crossview_process' and self.hparams.dataset_type == 'memory_depth_dji_instance_crossview':
                if self.hparams.dataset_type == 'memory_depth_dji_instance_crossview_process':
                    if item == ['end']:
                        print('done')
                        raise TypeError

                    continue
                
                if item == None:
                    continue
                elif item == ['end']:
                    self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index,
                                  dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)
                    if self.hparams.enable_instance and self.hparams.cached_centroids_type == 'all':
                        all_centroids = self.instance_cluster_prediction(train_iterations, dataset=dataset)
                        val_metrics, _ = self._run_validation(train_iterations, all_centroids)
                    else:
                        val_metrics, _ = self._run_validation(train_iterations)
                    self._write_final_metrics(val_metrics, train_iterations)
                    # raise TypeError

                

                # item = np.load('1236.npy',allow_pickle=True).item()

                if dataset_index <= discard_index:
                    continue
                discard_index = -1

                # amp: Automatic mixed precision
                with torch.cuda.amp.autocast(enabled=self.hparams.amp):
                    # 调整shape
                    if self.hparams.enable_semantic or self.hparams.enable_instance:
                        for key in item.keys():
                            if self.hparams.enable_instance:
                                if item[key].dim() == 2:
                                    item[key] = item[key].reshape(-1)
                                elif item[key].dim() == 3:
                                    item[key] = item[key].reshape(-1, *item[key].shape[2:])

                            if item[key].shape[0]==1:
                                item[key] = item[key].squeeze(0)
                            if item[key].dim() != 1:
                                if item[key].shape[-1] == 1:
                                    item[key] = item[key].reshape(-1)
                                else:
                                    item[key] = item[key].reshape(-1, item[key].shape[-1])
                                


                        for key in item.keys():
                            if 'random' in key:
                                continue
                            elif 'random_'+key in item.keys():
                                item[key] = torch.cat((item[key], item['random_'+key]))

                    # print(f"shape:{item['rgbs'].shape}")

                    if (self.hparams.enable_semantic or self.hparams.enable_instance) and 'labels' in item.keys():
                        labels = item['labels'].to(self.device, non_blocking=True)
                        if not self.hparams.enable_instance:
                            from tools.unetformer.uavid2rgb import remapping
                            labels = remapping(labels)
                    else:
                        labels = None

                    if 'sam' in self.hparams.dataset_type or self.hparams.dataset_type == 'mega_sa3d':
                        rgbs = None
                        if self.hparams.group_loss:
                            groups = item['groups'].to(self.device, non_blocking=True)
                        else:
                            groups = None
                    else:
                        rgbs = item['rgbs'].to(self.device, non_blocking=True)
                        groups = None

                    # training_step
                    metrics, bg_nerf_rays_present = self._training_step(
                        rgbs,
                        item['rays'].to(self.device, non_blocking=True),
                        item['img_indices'].to(self.device, non_blocking=True), 
                        labels, groups, train_iterations, item)
                    
                    if metrics == None and 'sa3d' in self.hparams.dataset_type:
                        continue

                    with torch.no_grad():
                        for key, val in metrics.items():
                            if key == 'psnr' and math.isinf(val):  # a perfect reproduction will give PSNR = infinity
                                continue

                            if not math.isfinite(val):
                                np.save(f"{train_iterations}.npy", item)
                                raise Exception('Train metrics not finite: {}'.format(metrics))
                            if math.isnan(val):
                                np.save(f"{train_iterations}.npy", item)
                                raise Exception('Train metrics is nan: {}'.format(metrics))

                for optimizer in optimizers.values():
                    # optimizer.zero_grad(set_to_none=True)
                    optimizer.zero_grad()

                scaler.scale(metrics['loss']).backward()   # 在这之后用torch.cuda.empty_cache() 有效

                if self.hparams.use_mask_type == 'densegrid_mlp':
                    with torch.no_grad():
                        self.nerf.seg_mask_grid.grid *= self.nerf.mask_view_counts
                        prev_mask_grid = self.nerf.seg_mask_grid.grid.detach().clone()


                if self.hparams.clip_grad_max != 0:
                    torch.nn.utils.clip_grad_norm_(self.nerf.parameters(), self.hparams.clip_grad_max)

                for key, optimizer in optimizers.items():
                    if key == 'bg_nerf' and (not bg_nerf_rays_present):
                        continue
                    else:
                        lr_temp = optimizer.param_groups[0]['lr']
                        if self.wandb is not None and train_iterations % self.hparams.logger_interval == 0:
                            self.wandb.log({"train/optimizer_{}_lr".format(key): lr_temp, 'epoch':train_iterations})
                        if self.writer is not None and train_iterations % self.hparams.logger_interval == 0:
                            self.writer.add_scalar('1_train/optimizer_{}_lr'.format(key), lr_temp, train_iterations)
                        scaler.step(optimizer)
                

                scaler.update()
                for scheduler in schedulers.values():
                    scheduler.step()

                if self.hparams.use_mask_type == 'densegrid_mlp':
                    with torch.no_grad():
                        self.nerf.mask_view_counts += (self.nerf.seg_mask_grid.grid != prev_mask_grid)
                        self.nerf.seg_mask_grid.grid /= (self.nerf.mask_view_counts + 1e-9)

                train_iterations += 1
                if self.is_master:
                    pbar.update(1)
                    if train_iterations % self.hparams.logger_interval == 0:
                        for key, value in metrics.items():
                            if self.writer is not None:
                                self.writer.add_scalar('1_train/{}'.format(key), value, train_iterations)
                            if self.wandb is not None:
                                self.wandb.log({"train/{}".format(key): value, 'epoch':train_iterations})

                    if train_iterations > 0 and train_iterations % self.hparams.ckpt_interval == 0:
                        self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index,
                                            dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)
            
                if (train_iterations > 0 and train_iterations % self.hparams.val_interval == 0) or train_iterations == self.hparams.train_iterations:
                    if self.hparams.enable_instance and self.hparams.cached_centroids_type == 'all':
                        all_centroids = self.instance_cluster_prediction(train_iterations, dataset=dataset)
                        val_metrics, all_centroids = self._run_validation(train_iterations, all_centroids)
                    else:
                        val_metrics, all_centroids = self._run_validation(train_iterations)
                        
                    
                    if 'llff' not in self.hparams.dataset_type and 'sa3d' not in self.hparams.dataset_type:
                        self._write_final_metrics(val_metrics, train_iterations)
                    
                    if self.hparams.enable_instance and train_iterations == self.hparams.train_iterations:
                        self.hparams.render_zyq = True 
                        if self.hparams.instance_loss_mode == 'linear_assignment':
                            all_centroids=None
                        self.hparams.fushi=True
                        _ = self._run_validation_render_zyq(train_iterations, all_centroids)
                
                if train_iterations >= self.hparams.train_iterations:
                    break
        

        if 'RANK' in os.environ:
            dist.barrier()

        if self.is_master:
            pbar.close()
            self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index,
                                  dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)


    def eval(self):

        if self.hparams.ckpt_path is not None:
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            train_iterations = checkpoint['iteration']
        self._setup_experiment_dir()

        if self.hparams.val_type == 'train':
            if self.hparams.enable_instance:
                with open(self.hparams.cached_centroids_path, 'rb') as f:
                    all_centroids = pickle.load(f)
                val_metrics, all_centroids = self._run_validation_train(train_iterations, all_centroids)
            else:
                val_metrics, _ = self._run_validation_train(train_iterations)
        elif self.hparams.val_type == 'train_instance':
            if self.hparams.enable_instance:
                if self.hparams.instance_loss_mode == 'linear_assignment':
                    all_centroids=None
                    val_metrics, _ = self._run_validation_instance(train_iterations)
                else:
                    with open(self.hparams.cached_centroids_path, 'rb') as f:
                        all_centroids = pickle.load(f)
                    val_metrics, all_centroids = self._run_validation_instance(train_iterations, all_centroids)
            else:
                val_metrics, _ = self._run_validation_instance(train_iterations)


        else:
            if self.hparams.enable_instance:
                all_centroids=None
                if self.hparams.cached_centroids_path is None:
                    if self.hparams.cached_centroids_type == 'all':
                        all_centroids = self.instance_cluster_prediction(train_iterations, dataset=None)
                else:
                    with open(self.hparams.cached_centroids_path, 'rb') as f:
                        all_centroids = pickle.load(f)
                val_metrics, all_centroids = self._run_validation(train_iterations, all_centroids)
            else:
                val_metrics, _ = self._run_validation(train_iterations)
            
            self._write_final_metrics(val_metrics, train_iterations=train_iterations)

            # 这里是渲染俯视图
            if self.hparams.enable_instance:
                self.hparams.render_zyq = True
                if self.hparams.instance_loss_mode == 'linear_assignment':
                    all_centroids=None
                self.hparams.fushi=True
                val_metrics = self._run_validation_render_zyq(train_iterations, all_centroids)
        


    def _write_final_metrics(self, val_metrics: Dict[str, float], train_iterations) -> None:
        if self.is_master:
            
            experiment_path_current = self.experiment_path / "eval_{}".format(train_iterations)
            with (experiment_path_current /'metrics.txt').open('a') as f:
                if 'pq' in val_metrics:
                    pq, sq, rq = val_metrics['pq'],val_metrics['sq'],val_metrics['rq']
                    print(f'pq, sq, rq: {pq:.5f} {sq:.5f} {rq:.5f}')

                    f.write(f'\n pq, sq, rq: {pq:.5f} {sq:.5f} {rq:.5f}\n')  
                    self.writer.add_scalar('2_val_metric_average/pq', pq, train_iterations)
                    self.writer.add_scalar('2_val_metric_average/sq', sq, train_iterations)
                    self.writer.add_scalar('2_val_metric_average/rq', rq, train_iterations)
                    
                    if 'all_centroids_shape' in val_metrics:
                        all_centroids_shape = val_metrics['all_centroids_shape']
                        self.writer.add_scalar('2_val_metric_average/all_centroids_shape', all_centroids_shape[0], train_iterations)
                        f.write('all_centroids_shape: {}\n'.format(all_centroids_shape))
                        del val_metrics['all_centroids_shape']

                    
                    metrics_each = val_metrics['metrics_each']
                    # f.write(f'panoptic metrics_each: {metrics_each} \n')  
                    
                    for key in metrics_each['all']:
                        avg_val = metrics_each['all'][key]
                        message = '      {}: {}'.format(key, avg_val)
                        f.write('{}\n'.format(message))
                        print(message)
                    f.write('{}\n')
                    f.write(f"pq, rq, sq, mIoU, TP, FP, FN: {metrics_each['all']['pq'][0].item()},{metrics_each['all']['rq'][0].item()},{metrics_each['all']['sq'][0].item()},{metrics_each['all']['iou_sum'][0].item()},{metrics_each['all']['true_positives'][0].item()},{metrics_each['all']['false_positives'][0].item()},{metrics_each['all']['false_negatives'][0].item()}\n")
                    del val_metrics['pq'],val_metrics['sq'],val_metrics['rq'], val_metrics['metrics_each']
                
                for key in val_metrics:
                    avg_val = val_metrics[key] / len(self.val_items)
                    if key== 'val/psnr':
                        if self.wandb is not None:
                            self.wandb.log({'val/psnr_avg': avg_val, 'epoch':train_iterations})
                        if self.writer is not None:
                            self.writer.add_scalar('2_val_metric_average/psnr_avg', avg_val, train_iterations)
                    if key== 'val/ssim':
                        if self.wandb is not None:
                            self.wandb.log({'val/ssim_avg': avg_val, 'epoch':train_iterations})
                        if self.writer is not None:
                            self.writer.add_scalar('2_val_metric_average/ssim_avg', avg_val, train_iterations)

                    message = 'Average {}: {}'.format(key, avg_val)
                    main_print(message)
                    f.write('{}\n'.format(message))
                psnr = val_metrics['val/psnr'] / len(self.val_items)
                ssim = val_metrics['val/ssim'] / len(self.val_items)
                abs_rel = val_metrics['val/abs_rel'] / len(self.val_items)
                rmse_actual = val_metrics['val/rmse_actual'] / len(self.val_items)
                if self.writer is not None:
                    self.writer.add_scalar('2_val_metric_average/abs_rel', abs_rel, train_iterations)
                    self.writer.add_scalar('2_val_metric_average/rmse_actual', rmse_actual, train_iterations)

                # f.write('arg_psnr, arg_ssim: {arg_psnr:.5f}, {arg_ssim:.5f}\n')  
                f.write(f'\n psnr, ssim, rmse_actual, abs_rel: {psnr:.5f}, {ssim:.5f}, {rmse_actual:.5f} ,{abs_rel:.5f}\n')  
                print(f'psnr, ssim, rmse_actual, abs_rel: {psnr:.5f}, {ssim:.5f}, {rmse_actual:.5f} ,{abs_rel:.5f}')

            self.writer.flush()
            self.writer.close()

    def instance_cluster_prediction(self, train_iterations, dataset=None):
        from gp_nerf.rendering_gpnerf import render_rays
        from tools.contrastive_lift.utils import create_instances_from_semantics

        # 先只考虑用聚类方式的instance
        if dataset == None:   # 在eval.py时不会读取data_loader,所以这里要手动读取
            from gp_nerf.datasets.memory_dataset_depth_dji_instance import MemoryDataset
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
            self.H = dataset.H
            self.W = dataset.W
        
        ##下面的代码对于 train.py 和 eval.py 都一样了
        #############  2023.10.23  dataset目前存了这些东西 
        # self._rgbs = rgbs
        # self._rays = rays
        # self._img_indices = indices  #  这个代码只存了一个
        # self._labels = instances
        # self._depth_djis = depth_djis 
        #############   render需要rays, gt_depth, image_indices
        ############# 10.23 为了方便， 把代码里的depth_scale在存入dataset时已经做处理，  从dataset里拿到的是相机到物体的距离（而不是z轴的分量）
        total_size = torch.cat(dataset._rays).shape[0]
        sample_size = int(3e7)
        sampled_indices = torch.randint(0, total_size, (sample_size,), dtype=torch.int64)

        rays = torch.cat(dataset._rays)[sampled_indices].to(self.device)
        instances = torch.cat(dataset._labels)[sampled_indices].to(self.device)
        image_indices = torch.ones_like(instances).to(self.device)
        gt_depths = torch.cat(dataset._depth_djis)[sampled_indices].to(self.device)

        results = {}
        block_size = self.hparams.image_pixel_batch_size*4
        with torch.no_grad():
            for i in tqdm(range(0, sampled_indices.shape[0], block_size),desc='sampling point to cluster instance'):

                result_batch, _ = render_rays(nerf=self.nerf, bg_nerf=self.bg_nerf,
                                                rays=rays[i:i + block_size],
                                                image_indices=image_indices[
                                                            i:i + block_size] if self.hparams.appearance_dim > 0 else None,
                                                hparams=self.hparams,
                                                sphere_center=self.sphere_center,
                                                sphere_radius=self.sphere_radius,
                                                get_depth=True,
                                                get_depth_variance=False,
                                                get_bg_fg_rgb=True,
                                                train_iterations=train_iterations,
                                                gt_depths= gt_depths[i:i + block_size] if gt_depths is not None else None,
                                                pose_scale_factor = self.pose_scale_factor)
                if 'air_sigma_loss' in result_batch:
                    del result_batch['air_sigma_loss']
                for key, value in result_batch.items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value.cpu())

        # for key, value in results.items():
        #     results[key] = torch.cat(value)

        # 以下对instance feature进行处理
        #################
        typ = 'fine'
        all_instance_features, all_thing_features = [], []

        for i in range(len(results[f'instance_map_{typ}'])):
            instances = results[f'instance_map_{typ}'][i]
            device = instances.device

            sem_logits = results[f'sem_map_{typ}'][i]
            sem_label = self.logits_2_label(sem_logits)
            sem_label = remapping(sem_label)
            
            if self.hparams.instance_loss_mode == 'slow_fast':
                slow_features = instances[...,self.hparams.num_instance_classes:] 
                # all_slow_features.append(slow_features)
                instances = instances[...,0:self.hparams.num_instance_classes] # keep fast features only

            p_instances = create_instances_from_semantics(instances, sem_label, self.thing_classes, device=device)
            all_instance_features.append(instances)
            all_thing_features.append(p_instances)
            #########################


        #  对所有计算的instance进行聚类
        # instance clustering
        all_instance_features = torch.cat(all_instance_features, dim=0).cpu().numpy()
        all_thing_features = torch.cat(all_thing_features, dim=0).cpu().numpy() # N x d
        
        experiment_path_current = self.experiment_path / "eval_{}".format(train_iterations)
        output_dir = str(experiment_path_current)
        if not os.path.exists(output_dir):
            Path(output_dir).mkdir(parents=True)
            
        # 下面的 all_points_instances直接返回None ,用不到
        all_points_instances, all_centroids = cluster(all_thing_features, bandwidth=0.2, device=self.device, use_dbscan=True)
        all_centroids_path = os.path.join(output_dir, f"all_centroids.npy")
        with open(all_centroids_path, "wb") as file:
            pickle.dump(all_centroids, file)
        print(f"save all_centroids_cache to : {all_centroids_path}")
        

        return all_centroids

    def _setup_experiment_dir(self) -> None:
        if self.is_master:
            self.experiment_path.mkdir()
            with (self.experiment_path / 'hparams.txt').open('w') as f:
                for key in vars(self.hparams):
                    f.write('{}: {}\n'.format(key, vars(self.hparams)[key]))
                if 'WORLD_SIZE' in os.environ:
                    f.write('WORLD_SIZE: {}\n'.format(os.environ['WORLD_SIZE']))

            with (self.experiment_path / 'command.txt').open('w') as f:
                f.write(' '.join(sys.argv))
                f.write('\n')

            self.model_path.mkdir(parents=True)

            if 'llff' not in self.hparams.dataset_type:
                with (self.experiment_path / 'image_indices.txt').open('w') as f:
                    for i, metadata_item in enumerate(self.train_items):
                        f.write('{},{}\n'.format(metadata_item.image_index, metadata_item.image_path.name))
        if self.hparams.writer_log:
            self.writer = SummaryWriter(str(self.experiment_path / 'tb')) if self.is_master else None
        if 'RANK' in os.environ:
            dist.barrier()
        
        if self.hparams.wandb_id =='None':
            self.wandb = None
            print('no using wandb')
        else:
            self.wandb = wandb.init(project=self.hparams.wandb_id, entity="mega-ingp", name=self.hparams.wandb_run_name, dir=self.experiment_path)
            
    def ema_update_slownet(self, slownet, fastnet, momentum):
        # EMA update for the teacher
        with torch.no_grad():
            for param_q, param_k in zip(fastnet.parameters(), slownet.parameters()):
                param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

    def _training_step(self, rgbs: torch.Tensor, rays: torch.Tensor, image_indices: Optional[torch.Tensor], labels: Optional[torch.Tensor], groups: Optional[torch.Tensor], train_iterations = -1, item=None) \
            -> Tuple[Dict[str, Union[torch.Tensor, float]], bool]:
        if 'sdf' in self.hparams.network_type:
            if self.hparams.sdf_as_gpnerf:
                from gp_nerf.rendering_gpnerf_clean_sdf_as_gpnerf import render_rays
            else:
                from gp_nerf.rendering_gpnerf_clean_sdf import render_rays
        else:
            from gp_nerf.rendering_gpnerf import render_rays
        
        if 'depth_dji' in item:
            gt_depths = item['depth_dji'].to(self.device, non_blocking=True)
        else:
            gt_depths = None

        if 'sa3d' not in self.hparams.dataset_type:
            results, bg_nerf_rays_present = render_rays(nerf=self.nerf,
                                                        bg_nerf=self.bg_nerf,
                                                        rays=rays,
                                                        image_indices=image_indices,
                                                        hparams=self.hparams,
                                                        sphere_center=self.sphere_center,
                                                        sphere_radius=self.sphere_radius,
                                                        get_depth=False,
                                                        get_depth_variance=True,
                                                        get_bg_fg_rgb=False,
                                                        train_iterations=train_iterations,
                                                        gt_depths=gt_depths,
                                                        pose_scale_factor = self.pose_scale_factor
                                                        )
        else:
            bg_nerf_rays_present=False
            results_chunks = []
            B = rays.shape[0]
            keys = ['rgb_fine', 'sem_map_fine', 'depth_variance_fine']
            for i in range(0, B, self.hparams.ray_chunk_size):
                # torch.cuda.empty_cache()
                ray_chunk = rays[i:i + self.hparams.ray_chunk_size]
                image_indices_chunk= image_indices[i:i + self.hparams.ray_chunk_size]
                results_chunk, _ = render_rays(nerf=self.nerf,
                                            bg_nerf=self.bg_nerf,
                                            rays=ray_chunk,
                                            image_indices=image_indices_chunk,
                                            hparams=self.hparams,
                                            sphere_center=self.sphere_center,
                                            sphere_radius=self.sphere_radius,
                                            get_depth=False,
                                            get_depth_variance=True,
                                            get_bg_fg_rgb=False,
                                            train_iterations=train_iterations
                                            )
                
                results_chunk_dict = {k: v for k, v in results_chunk.items()}
                results_chunks += [results_chunk_dict]
                del results_chunk, results_chunk_dict
                # gc.collect()
                # torch.cuda.empty_cache()
            
            H, W = self.H, self.W
            results = {
                k: torch.cat([ret[k] for ret in results_chunks])#.reshape(H,W,-1)
                for k in results_chunks[0].keys()
            }
            
        
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        if not self.hparams.freeze_geo and not ('sam' in self.hparams.dataset_type):
            if 'sdf' in self.hparams.network_type:
                with torch.no_grad():
                    psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
                    # depth_variance = results[f'depth_variance_{typ}'].mean()
                metrics = {
                    'psnr': psnr_,
                    # 'depth_variance': depth_variance,
                }
            else:
                with torch.no_grad():
                    psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
                    depth_variance = results[f'depth_variance_{typ}'].mean()
                metrics = {
                    'psnr': psnr_,
                    'depth_variance': depth_variance,
                }
            metrics['loss'] = 0

            photo_loss = F.mse_loss(results[f'rgb_{typ}'], rgbs, reduction='mean')
            metrics['photo_loss'] = photo_loss
            metrics['loss'] += photo_loss
        else:
            metrics = {}
            metrics['loss'] = 0
            photo_loss = torch.zeros(1, device='cuda')
            metrics['photo_loss'] = photo_loss
            metrics['loss'] += photo_loss

        if 'air_sigma_loss' in results:
            air_sigma_loss = results['air_sigma_loss'] * self.hparams.wgt_air_sigma_loss
            metrics['air_sigma_loss'] = air_sigma_loss
            metrics['loss'] +=  air_sigma_loss


        # depth_dji loss
        if self.hparams.depth_dji_loss:
            valid_depth_mask = ~torch.isinf(gt_depths)
            pred_depths = results['depth_fine']
            pred_depths_valid = pred_depths[valid_depth_mask]
            gt_depths_valid = gt_depths[valid_depth_mask]

            if self.hparams.wgt_depth_mse_loss != 0:
                depth_mse_loss = torch.mean((pred_depths_valid - gt_depths_valid)**2)
                metrics['depth_mse_loss'] = depth_mse_loss
                metrics['loss'] += self.hparams.wgt_depth_mse_loss * depth_mse_loss

            if self.hparams.wgt_sigma_loss != 0:
                ### sigma_loss from dsnerf (Ray distribution loss): add additional depth supervision
                # z_vals = results['zvals_fine'][valid_depth_mask]
                # deltas = results['deltas_fine'][valid_depth_mask]
                # weights = results[f'weights_fine'][valid_depth_mask]
                # rays_d = rays[valid_depth_mask, 3:6]
                # err = 1
                # dists = deltas * torch.norm(rays_d[...,None,:], dim=-1)
                # sigma_loss = -torch.log(weights + 1e-5) * torch.exp(-(z_vals - gt_depths_valid[:,None]) ** 2 / (2 * err)) * dists
                # sigma_loss = torch.sum(sigma_loss, dim=1).mean()
                # metrics['sigma_loss'] = sigma_loss

                metrics['sigma_loss'] = results['sigma_loss']
                metrics['loss'] += self.hparams.wgt_sigma_loss * sigma_loss
                
        # instance loss
        if self.hparams.enable_instance:
            instance_features = results[f'instance_map_{typ}']
            labels_gt = labels.type(torch.long)
            if 'InstanceBuilding' not in self.hparams.dataset_path:
                sem_logits = results[f'sem_map_{typ}']
                sem_label = self.logits_2_label(sem_logits)
                sem_label = remapping(sem_label)
            else:
                sem_label = item['semantic_labels']


            # contrastive loss or slow-fast loss
            instance_loss, concentration_loss = self.calculate_instance_clustering_loss(instance_features, labels_gt)

            # Concentration loss from contrastive lift

            metrics['instance_loss'] = instance_loss

            metrics['concentration_loss'] = concentration_loss

            metrics['loss'] += self.hparams.wgt_instance_loss * instance_loss + self.hparams.wgt_concentration_loss * concentration_loss

        #semantic loss
        if self.hparams.enable_semantic and (not self.hparams.freeze_semantic):
            if 'sa3d' in self.hparams.dataset_type:
                sem_logits = results[f'sem_map_{typ}']
                if self.hparams.use_mask_type == 'hashgrid_mlp':
                    sem_logits = self.nerf.mask_fc_hash(sem_logits)
                elif self.hparams.use_mask_type == 'densegrid_mlp':
                    sem_logits = self.nerf.mask_fc_dense(sem_logits)
                 
                if labels is not None:  # 第一祯
                    # print(sem_logits.unique())
                    loss_sam = 0
                    # print(sem_logits.max())
                    for seg_idx in range(sem_logits.shape[-1]):
                        loss_sam += seg_loss(labels[:, seg_idx].view(H, W), None, sem_logits[:, seg_idx:seg_idx+1].view(H, W, 1))
                    # cv2.imwrite("00001.jpg", (labels[:, 0]>0).view(H, W, 1).repeat(1,1,3).cpu().numpy()*255)

                    ###  查看第一祯的监督，正常（从grid训练正常也能判断）
                    # tmp_mask = np.zeros((H,W,3))
                    # selected_points = item['selected_points']
                    # for index_c in range(len(selected_points)):
                    #     x, y = selected_points[index_c, 0], selected_points[index_c, 1]
                    #     tmp_mask[x, y] = [0, 0, 255] if labels[index_c]==1 else [0, 255, 0]

                else:   #其他祯

                    if self.hparams.sa3d_whole_image:
                        sem_logits = sem_logits.view(H, W, -1)
                        if sem_logits.max() < 0:
                            print('There is no positive value in sem_logits')
                            return None, None
                        # cv2.imwrite("00001.jpg", (sem_logits>0).repeat(1,1,3).cpu().numpy()*255)
                        depth = item['depth'].view(H, W, 1)
                        # set feature
                        sam_feature = item['sam_feature'].squeeze(0).to(self.device)
                        self.predictor.set_feature(sam_feature, [H, W])
                        # set_image
                        # init_image = item['rgbs'].view(H, W, -1)
                        # init_image = to8b(init_image.cpu().numpy())
                        # self.predictor.set_image(init_image)
                        
                        # a = torch.HalfTensor(np.load('/data/yuqi/code/SegmentAnythingin3D/depth/0001.npy'))
                        # b= _generate_index_matrix(H, W, a.detach().clone())
                        # loss_sam, sam_seg_show = prompting_coarse(self, H, W, sem_logits, b.to(self.device), self.hparams.num_semantic_classes)    

                        index_matrix = _generate_index_matrix(H, W, depth.detach().clone())  # 【H,W,3】分别存储的是x y depth
                        loss_sam, sam_seg_show = prompting_coarse(self, H, W, sem_logits, index_matrix.to(self.device), self.hparams.num_semantic_classes)    
                    else:
                        sem_logits = results[f'sem_map_{typ}']
                        depth = item['depth'].unsqueeze(-1)  # H * W * 1
                        sam_feature = item['sam_feature'].squeeze(0).to(self.device)
                        self.predictor.set_feature(sam_feature, [H, W])
                        index_matrix = _generate_index_matrix(H, W, depth.detach().clone())  # 【H,W,3】分别存储的是x y depth
                        selected_points = item['selected_points']
                        loss_sam, sam_seg_show = prompting_coarse_N(self, H, W, sem_logits, index_matrix, self.hparams.num_semantic_classes, selected_points)
                    if loss_sam == 0:
                        return None, None

                metrics['loss_sam'] = loss_sam
                metrics['loss'] += self.hparams.wgt_sam_loss * loss_sam
            
            else:
                sem_logits = results[f'sem_map_{typ}']
                semantic_loss = self.crossentropy_loss(sem_logits, labels.type(torch.long))
                metrics['semantic_loss'] = semantic_loss
                metrics['loss'] += self.hparams.wgt_sem_loss * semantic_loss
                
                with torch.no_grad():
                    if train_iterations % 1000 == 0:
                        sem_label = self.logits_2_label(sem_logits)
                        if self.writer is not None:
                            self.writer.add_scalar('1_train/accuracy', sum(labels == sem_label) / labels.shape[0], train_iterations)
                        if self.wandb is not None:
                            self.wandb.log({'train/accuracy': sum(labels == sem_label) / labels.shape[0], 'epoch': train_iterations})

                if self.hparams.dataset_type == 'sam':
                    if self.hparams.group_loss:
                        # group loss
                        semantic_feature  = results[f'semantic_feature_{typ}']
                        group_ids, group_counts = torch.unique(groups, return_counts=True)
                        group_loss_each = 0
                        for group_id in group_ids:
                            group_index = (groups==group_id)
                            feature_group = semantic_feature[group_index]
                            f_detach = feature_group.detach()
                            feature_group_mean = f_detach.mean(dim=0).repeat(feature_group.shape[0],1)
                            cs_loss = self.group_loss_feat(feature_group, feature_group_mean)
                            group_loss_each += cs_loss.mean()
                        group_loss = 1 - group_loss_each / len(group_ids) #/ semantic_feature.shape[-1]
                        metrics['sam_group_loss'] = group_loss
                        metrics['loss'] += self.hparams.wgt_group_loss * group_loss
            


        if 'sdf' in self.hparams.network_type and not self.hparams.freeze_geo:

            # sdf 
            metrics['gradient_error'] = results[f'gradient_error_{typ}'].squeeze(-1)
            if self.hparams.gradient_error_weight_increase and train_iterations > self.hparams.train_iterations / 2:
                gradient_error_weight = 0.1
            else:
                gradient_error_weight = self.hparams.gradient_error_weight
            metrics['loss'] += gradient_error_weight * results[f'gradient_error_{typ}'].squeeze(-1)
            if f'curvature_error_{typ}' in results:
                metrics['curvature_error'] = results[f'curvature_error_{typ}'].squeeze(-1)
                metrics['loss'] += 0.1 * results[f'curvature_error_{typ}'].squeeze(-1)
            if self.wandb is not None:
                self.wandb.log({"train/inv_s": 1.0 / results['inv_s'], 'epoch': train_iterations})
            if self.writer is not None:
                self.writer.add_scalar('1_train/inv_s', 1.0 / results['inv_s'], train_iterations)
            if self.writer is not None:
                self.writer.add_scalar('1_train/cos_anneal_ratio', results['cos_anneal_ratio'], train_iterations)
            if self.writer is not None:
                self.writer.add_scalar('1_train/normal_epsilon_ratio', results['normal_epsilon_ratio'], train_iterations)
            del results['cos_anneal_ratio'], results['normal_epsilon_ratio']
        return metrics, bg_nerf_rays_present
    
    def create_virtual_gt_with_linear_assignment(self, labels_gt, predicted_scores):
        labels = sorted(torch.unique(labels_gt).cpu().tolist())[:predicted_scores.shape[-1]]
        predicted_probabilities = torch.softmax(predicted_scores, dim=-1).detach()
        cost_matrix = np.zeros([len(labels), predicted_probabilities.shape[-1]])
        for lidx, label in enumerate(labels):
            cost_matrix[lidx, :] = -(predicted_probabilities[labels_gt == label, :].sum(dim=0) / ((labels_gt == label).sum() + 1e-4)).cpu().numpy()
        assignment = scipy.optimize.linear_sum_assignment(np.nan_to_num(cost_matrix))
        new_labels = torch.zeros_like(labels_gt)
        for aidx, lidx in enumerate(assignment[0]):
            new_labels[labels_gt == labels[lidx]] = assignment[1][aidx]
        return new_labels
    
    def calculate_instance_clustering_loss(self, instance_features, labels_gt):
        instance_loss = 0    #torch.tensor(0., device=instance_features.device, requires_grad=True)
        concentration_loss = 0 ##torch.tensor(0., device=instance_features.device, requires_grad=True)
        if instance_features == []:
            return torch.tensor(0., device=instance_features.device), torch.tensor(0., device=instance_features.device)
        if self.hparams.instance_loss_mode == "linear_assignment":
            # 2023.10.31 18:54
            virtual_gt_labels = self.create_virtual_gt_with_linear_assignment(labels_gt, instance_features)
            predicted_labels = instance_features.argmax(dim=-1)
            if torch.any(virtual_gt_labels != predicted_labels):  # should never reinforce correct labels
                # return (self.loss_instances_cluster(instance_features, virtual_gt_labels) * confidences).mean()
                # 我们这里先不考虑confidences
                return (self.loss_instances_cluster(instance_features, virtual_gt_labels)).mean(), concentration_loss
            return torch.tensor(0., device=instance_features.device, requires_grad=True), concentration_loss
        
        elif self.hparams.instance_loss_mode == "contrastive": # vanilla contrastive loss
            instance_loss = contrastive_loss(instance_features, labels_gt, self.temperature)
        
        elif self.hparams.instance_loss_mode == "slow_fast":    
            # EMA update of slow network; done before everything else
            ema_momentum = 0.9 # CONSTANT MOMENTUM
            
            self.ema_update_slownet(self.nerf.instance_linear_slow, self.nerf.instance_linear, ema_momentum)
            self.ema_update_slownet(self.nerf.instance_linear_slow_bg, self.nerf.instance_linear_bg, ema_momentum)

            fast_features, slow_features = instance_features.split(
                [self.hparams.num_instance_classes, self.hparams.num_instance_classes], dim=-1)
            
            fast_projections, slow_projections = fast_features, slow_features # no projection layer
            slow_projections = slow_projections.detach() # no gradient for slow projections

            # sample two random batches from the current batch
            fast_mask = torch.zeros_like(labels_gt).bool()
            # 1026 这里采用固定的后半段， 改为随机一半
            # fast_mask[:labels_gt.shape[0] // 2] = True
            random_indices = torch.randperm(len(labels_gt))[:len(labels_gt) // 2]
            fast_mask[random_indices] = True
            slow_mask = ~fast_mask # non-overlapping masks for slow and fast models
            ## compute centroids
            slow_centroids = []
            fast_labels, slow_labels = torch.unique(labels_gt[fast_mask]), torch.unique(labels_gt[slow_mask])
            for l in slow_labels:
                mask_ = torch.logical_and(slow_mask, labels_gt==l) #.unsqueeze(-1)
                slow_centroids.append(slow_projections[mask_].mean(dim=0))
            slow_centroids = torch.stack(slow_centroids)
            # DEBUG edge case:
            if len(fast_labels) == 0 or len(slow_labels) == 0:
                print("Length of fast labels", len(fast_labels), "Length of slow labels", len(slow_labels))
                # This happens when labels_gt of shape 1
                return torch.tensor(0.0, device=instance_features.device),torch.tensor(0.0, device=instance_features.device)
            
            ### Concentration loss
            intersecting_labels = fast_labels[torch.where(torch.isin(fast_labels, slow_labels))] # [num_centroids]
            for l in intersecting_labels:
                mask_ = torch.logical_and(fast_mask, labels_gt==l)
                centroid_ = slow_centroids[slow_labels==l] # [1, d]
                # distance between fast features and slow centroid
                dist_sq = torch.pow(fast_projections[mask_] - centroid_, 2).sum(dim=-1) # [num_points]
                # loss += -1.0 * (torch.exp(-dist_sq / 1.0) * confidences[mask_]).mean()  # 1024 暂时不考虑confidence
                concentration_loss += -1.0 * (torch.exp(-dist_sq / 1.0)).mean()  # 1024 暂时不考虑confidence

            if intersecting_labels.shape[0] > 0: 
                concentration_loss /= intersecting_labels.shape[0]
            
            ### Contrastive loss
            label_matrix = labels_gt[fast_mask].unsqueeze(1) == labels_gt[slow_mask].unsqueeze(0) # [num_points1, num_points2]
            similarity_matrix = torch.exp(-torch.cdist(fast_projections[fast_mask], slow_projections[slow_mask], p=2) / 1.0) # [num_points1, num_points2]
            logits = torch.exp(similarity_matrix)
            # compute loss
            prob = torch.mul(logits, label_matrix).sum(dim=-1) / logits.sum(dim=-1)
            prob_masked = torch.masked_select(prob, prob.ne(0))
            if prob_masked.shape[0] == 0:
                return torch.tensor(0.0, device=instance_features.device, requires_grad=True),torch.tensor(0.0, device=instance_features.device, requires_grad=True)
            instance_loss += -torch.log(prob_masked).mean()

        return instance_loss, concentration_loss

    def render_zyq(self):
        if self.hparams.ckpt_path is not None:
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            train_iterations = checkpoint['iteration']
        self._setup_experiment_dir()

        if self.hparams.enable_instance:
            if self.hparams.instance_loss_mode != 'linear_assignment':
                with open(self.hparams.cached_centroids_path, 'rb') as f:
                    all_centroids = pickle.load(f)
            else:
                all_centroids=None
            val_metrics = self._run_validation_render_zyq(train_iterations, all_centroids)
        else:
            val_metrics = self._run_validation_render_zyq(train_iterations)

        self._write_final_metrics(val_metrics, train_iterations=train_iterations)
    
    def _run_validation_render_zyq(self, train_index=-1, all_centroids=None) -> Dict[str, float]:

        with torch.inference_mode():
            #semantic 
            self.metrics_val = Evaluator(num_class=self.hparams.num_semantic_classes)
            CLASSES = ('Cluster', 'Building', 'Road', 'Car', 'Tree', 'Vegetation', 'Human', 'Sky', 'Water', 'Ground', 'Mountain')
            self.nerf.eval()
            val_metrics = defaultdict(float)
            base_tmp_path = None
            
            dataset_path = Path(self.hparams.dataset_path)

            if self.hparams.fushi:
                    
                if 'Yingrenshi' in self.hparams.dataset_path:
                    val_paths = sorted(list((dataset_path / 'render_far0.3' / 'metadata').iterdir()))
                else:
                    val_paths = sorted(list((dataset_path / 'render_far0.3_val' / 'metadata').iterdir()))
            else:
                # val_paths = sorted(list((dataset_path / 'render_supp' / 'metadata').iterdir()))
                #####  2024/01/20   demo 改的，  上面是补充材料时候用的
                # val_paths = sorted(list((dataset_path / 'render_far0.3' / 'metadata').iterdir()))
                #####  2024/01/25   cvpr rebuttal
                # val_paths = sorted(list((dataset_path / 'render_far0.3_val' / 'metadata').iterdir()))
                val_paths = sorted(list((dataset_path / self.hparams.render_zyq_far_view / 'metadata').iterdir()))



            train_paths = val_paths
            train_paths.sort(key=lambda x: x.name)
            val_paths_set = set(val_paths)
            image_indices = {}
            for i, train_path in enumerate(train_paths):
                image_indices[train_path.name] = i
            render_items = [
                self._get_metadata_item(x, image_indices[x.name], self.hparams.val_scale_factor, x in val_paths_set) for x
                in train_paths]

            H = render_items[0].H
            W = render_items[0].W

            if self.hparams.start == -1 and self.hparams.end == -1:
                indices_to_eval = np.arange(len(render_items))
            else:
                indices_to_eval = np.arange(len(render_items))[self.hparams.start:self.hparams.end]
            

            # used_files = []
            # for ext in ('*.png', '*.jpg'):
            #     used_files.extend(glob(os.path.join('/data/yuqi/Datasets/DJI/Yingrenshi_20230926_subset/train/rgbs', ext)))
            # used_files.sort()
            # process_item = [Path(far_p).stem for far_p in used_files]

            if self.hparams.enable_instance and self.hparams.render_zyq and self.hparams.fushi:
                experiment_path_current = self.experiment_path / "eval_fushi"
            else:
                experiment_path_current = self.experiment_path / "eval_{}".format(train_index)
            Path(str(experiment_path_current)).mkdir()
            Path(str(experiment_path_current / 'val_rgbs')).mkdir()
            Path(str(experiment_path_current / 'val_rgbs'/'pred_all')).mkdir()
            with (experiment_path_current / 'psnr.txt').open('w') as f:
                
                samantic_each_value = {}
                for class_name in CLASSES:
                    samantic_each_value[f'{class_name}_iou'] = []
                samantic_each_value['mIoU'] = []
                samantic_each_value['FW_IoU'] = []
                samantic_each_value['F1'] = []
                
                if self.hparams.enable_instance:
                    all_instance_features, all_thing_features = [], []
                    all_points_rgb, all_points_semantics = [], []
                    gt_points_rgb, gt_points_semantic, gt_points_instance = [], [], []
                    if self.hparams.fushi:
                        if 'Yingrenshi' in self.hparams.dataset_path:
                            indices_to_eval = indices_to_eval[:1]
                        elif 'Longhua_block2' in self.hparams.dataset_path:
                            indices_to_eval = indices_to_eval[4:5]
                        elif 'Longhua_block1' in self.hparams.dataset_path:
                            indices_to_eval = indices_to_eval[5:6]
                        elif 'Campus_new' in self.hparams.dataset_path:
                            # indices_to_eval = indices_to_eval[3:4]
                            indices_to_eval = indices_to_eval[7:8]
                            # indices_to_eval = indices_to_eval[7:8]

                
                for i in main_tqdm(indices_to_eval):
                    self.metrics_val_each = Evaluator(num_class=self.hparams.num_semantic_classes)
                    metadata_item = render_items[i]

                    # file_name = Path(metadata_item.image_path).stem
                    # if file_name not in process_item:
                    #     continue
                    i = int(Path(metadata_item.depth_dji_path).stem)
                    # i = metadata_item.image_index
                    self.hparams.sampling_mesh_guidance = False
                    results, _ = self.render_image(metadata_item, train_index)
                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    
                    # get rendering rgbs and depth
                    viz_result_rgbs = results[f'rgb_{typ}'].view(H, W, 3).cpu()
                    viz_result_rgbs = viz_result_rgbs.clamp(0,1)
                    self.H, self.W = viz_result_rgbs.shape[0],viz_result_rgbs.shape[1]

                    save_depth_dir = os.path.join(str(experiment_path_current), 'val_rgbs', "pred_depth_save")
                    if not os.path.exists(save_depth_dir):
                        os.makedirs(save_depth_dir)
                    depth_map = results[f'depth_{typ}'].view(viz_result_rgbs.shape[0], viz_result_rgbs.shape[1]).numpy().astype(np.float16)
                    np.save(os.path.join(save_depth_dir, ("%06d.npy" % i)), depth_map)

                    if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'pred_rgb')):
                        Path(str(experiment_path_current / 'val_rgbs' / 'pred_rgb')).mkdir()
                    Image.fromarray((viz_result_rgbs.numpy() * 255).astype(np.uint8)).save(
                        str(experiment_path_current / 'val_rgbs' / 'pred_rgb' / ("%06d.jpg" % i)))
                    
                    
                     
                    # NOTE: 这里初始化了一个list，需要可视化的东西可以后续加上去
                    img_list = [viz_result_rgbs * 255]

                    
                    prepare_depth_normal_visual(img_list, self.hparams, metadata_item, typ, results, Runner.visualize_scalars, experiment_path_current, i)

                    get_semantic_gt_pred_render_zyq(results, 'val', metadata_item, viz_result_rgbs, self.logits_2_label, typ, remapping,
                                        self.metrics_val, self.metrics_val_each, img_list, experiment_path_current, i, self.writer, self.hparams)
                    
                    if self.hparams.enable_instance:
                        instances, p_instances, all_points_rgb, all_points_semantics, gt_points_semantic, p_instances_building = get_instance_pred(
                                        results, 'val', metadata_item, viz_result_rgbs, self.logits_2_label, typ, remapping, 
                                        experiment_path_current, i, self.writer, self.hparams, viz_result_rgbs, self.thing_classes,
                                        all_points_rgb, all_points_semantics, gt_points_semantic)
            
                        all_instance_features.append(instances)
                        all_thing_features.append(p_instances)


                    # NOTE: 对需要可视化的list进行处理
                    # save images: list：  N * (H, W, 3),  -> tensor(N, 3, H, W)
                    # 将None元素转换为zeros矩阵
                    img_list = [torch.zeros_like(viz_result_rgbs) if element is None else element for element in img_list]
                    img_list = torch.stack(img_list).permute(0,3,1,2)
                    img = make_grid(img_list, nrow=3)
                    img_grid = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    Image.fromarray(img_grid).save(str(experiment_path_current / 'val_rgbs' / 'pred_all'/ ("%06d_all.jpg" % i)))

                    del results

            if self.hparams.enable_instance:
                if self.hparams.instance_loss_mode == 'linear_assignment':
                    all_points_instances = torch.stack(all_thing_features, dim=0) # N x d
                    output_dir = str(experiment_path_current / 'panoptic')
                    if not os.path.exists(output_dir):
                        Path(output_dir).mkdir()
                else:
                    # instance clustering
                    all_instance_features = torch.cat(all_instance_features, dim=0).cpu().numpy()
                    all_thing_features = torch.cat(all_thing_features, dim=0).cpu().numpy() # N x d
                    output_dir = str(experiment_path_current / 'panoptic')
                    if not os.path.exists(output_dir):
                        Path(output_dir).mkdir()
                    # np.save(os.path.join(output_dir, "all_thing_features.npy"), all_thing_features)
                    # np.save(os.path.join(output_dir, "all_points_semantics.npy"), torch.stack(all_points_semantics).cpu().numpy())
                    # np.save(os.path.join(output_dir, "all_points_rgb.npy"), torch.stack(all_points_rgb).cpu().numpy())
                    if self.hparams.cached_centroids_type == 'all':
                        all_points_instances = assign_clusters(all_thing_features, all_points_semantics, all_centroids, 
                                                                device=self.device, num_images=len(indices_to_eval))
                    elif self.hparams.cached_centroids_type == 'test':
                        all_points_instances, all_centroids = cluster(all_thing_features, bandwidth=0.2, device=self.device, 
                                                    num_images=len(indices_to_eval), use_dbscan=self.hparams.use_dbscan, all_centroids=all_centroids)

                if not os.path.exists(str(experiment_path_current / 'pred_semantics')):
                    Path(str(experiment_path_current / 'pred_semantics')).mkdir()
                if not os.path.exists(str(experiment_path_current / 'pred_surrogateid')):
                    Path(str(experiment_path_current / 'pred_surrogateid')).mkdir()

                for save_i in range(len(indices_to_eval)):
                    p_rgb = all_points_rgb[save_i]
                    p_semantics = all_points_semantics[save_i]
                    p_instances = all_points_instances[save_i]

                    
                    output_semantics_with_invalid = p_semantics.detach()
                    Image.fromarray(output_semantics_with_invalid.reshape(self.H, self.W).cpu().numpy().astype(np.uint8)).save(
                            str(experiment_path_current / 'pred_semantics'/ ("%06d.png" % int(self.val_items[i].image_path.stem))))
                    
                    Image.fromarray(p_instances.argmax(dim=1).reshape(self.H, self.W).cpu().numpy().astype(np.uint16)).save(
                            str(experiment_path_current / 'pred_surrogateid'/ ("%06d.png" % int(self.val_items[i].image_path.stem))))
                    
                    stack = visualize_panoptic_outputs(
                        p_rgb, p_semantics, p_instances, None, None, None, None,
                        self.H, self.W, thing_classes=self.thing_classes, visualize_entropy=False
                    )
                    grid = make_grid(stack, value_range=(0, 1), normalize=True, nrow=3).permute((1, 2, 0)).contiguous()
                    grid = (grid * 255).cpu().numpy().astype(np.uint8)
                    
                    Image.fromarray(grid).save(str(experiment_path_current / 'panoptic' / ("%06d.jpg" % save_i)))

            return val_metrics
            
    
    def _run_validation(self, train_index=-1, all_centroids=None) -> Dict[str, float]:
        from tools.unetformer.uavid2rgb import remapping
        with torch.inference_mode():
            #semantic 
            self.metrics_val = Evaluator(num_class=self.hparams.num_semantic_classes)
            CLASSES = ('Cluster', 'Building', 'Road', 'Car', 'Tree', 'Vegetation', 'Human', 'Sky', 'Water', 'Ground', 'Mountain')
            self.nerf.eval()
            val_metrics = defaultdict(float)
            base_tmp_path = None
            
            val_type = self.hparams.val_type  # train  val
            print('val_type: ', val_type)
            try:
                if val_type == 'val':
                    if 'residence'in self.hparams.dataset_path:
                        self.val_items=self.val_items[:19]
                    # elif 'building'in self.hparams.dataset_path or 'campus'in self.hparams.dataset_path:
                        # self.val_items=self.val_items[:10]
                    indices_to_eval = np.arange(len(self.val_items))
                elif 'train' in val_type:
                    indices_to_eval = np.arange(len(self.train_items))
                    print(len(self.train_items))
                    # indices_to_eval = np.arange(370,490)  
                    
                if self.hparams.enable_instance:
                    all_instance_features, all_thing_features = [], []
                    all_thing_features_building = []
                    all_points_rgb, all_points_semantics = [], []
                    gt_points_rgb, gt_points_semantic, gt_points_instance = [], [], []
                experiment_path_current = self.experiment_path / "eval_{}".format(train_index)
                if self.hparams.cached_centroids_path is None and self.hparams.enable_instance:
                    Path(str(experiment_path_current)).mkdir(exist_ok=True)
                    Path(str(experiment_path_current / 'val_rgbs')).mkdir(exist_ok=True)
                else:
                    Path(str(experiment_path_current)).mkdir()
                    Path(str(experiment_path_current / 'val_rgbs')).mkdir()
                with (experiment_path_current / 'psnr.txt').open('w') as f:
                    
                    samantic_each_value = {}
                    for class_name in CLASSES:
                        samantic_each_value[f'{class_name}_iou'] = []
                    samantic_each_value['mIoU'] = []
                    samantic_each_value['FW_IoU'] = []
                    samantic_each_value['F1'] = []
                    # samantic_each_value['OA'] = []

                    
                    if self.hparams.debug:
                        indices_to_eval = indices_to_eval[:2]
                        # indices_to_eval = indices_to_eval[:2]
                    
                    for i in main_tqdm(indices_to_eval):
                        self.metrics_val_each = Evaluator(num_class=self.hparams.num_semantic_classes)
                        # if i != 0:
                        #     break
                        if val_type == 'val':
                            metadata_item = self.val_items[i]
                        elif 'train' in val_type:
                            metadata_item = self.train_items[i]
                            if metadata_item.is_val:
                                continue

                        if self.hparams.enable_instance:
                            gt_instance_label = metadata_item.load_instance_gt()
                            gt_points_instance.append(gt_instance_label.view(-1))
                        
                        if self.hparams.eval_others:

                            eval_others(val_type, metadata_item,remapping,self.hparams, self.metrics_val, self.metrics_val_each)
                            continue

                        results, _ = self.render_image(metadata_item, train_index)


                        typ = 'fine' if 'rgb_fine' in results else 'coarse'
                        
                        viz_rgbs = metadata_item.load_image().float() / 255.
                        self.H, self.W = viz_rgbs.shape[0], viz_rgbs.shape[1]
                        if self.hparams.save_depth:
                            save_depth_dir = os.path.join(str(self.experiment_path), "depth_{}".format(train_index))
                            if not os.path.exists(save_depth_dir):
                                os.makedirs(save_depth_dir)
                            depth_map = results[f'depth_{typ}'].view(viz_rgbs.shape[0], viz_rgbs.shape[1]).numpy().astype(np.float16)
                            np.save(os.path.join(save_depth_dir, metadata_item.image_path.stem + '.npy'), depth_map)
                            continue
                        
                        # get rendering rgbs and depth
                        viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()

                        if self.hparams.eval_others and self.hparams.eval_others_name == 'labels_panolift':
                            viz_result_rgbs = Image.open(os.path.join(self.hparams.dataset_path, 'val', 'rgbs_panolift',(Path(metadata_item.image_path).stem+'.png'))).convert('RGB')
                            size = viz_result_rgbs.size
                            if size[0] != self.W or size[1] != self.H:
                                viz_result_rgbs = viz_result_rgbs.resize((self.W, self.H), Image.BILINEAR)
                            viz_result_rgbs = torch.tensor(np.asarray(viz_result_rgbs))/255


                        viz_result_rgbs = viz_result_rgbs.clamp(0,1)
                        if val_type == 'val':   # calculate psnr  ssim  lpips when val (not train)
                            val_metrics = calculate_metric_rendering(viz_rgbs, viz_result_rgbs, train_index, self.wandb, self.writer, val_metrics, i, f, self.hparams, metadata_item, typ, results, self.device, self.pose_scale_factor)
                        
                        viz_result_rgbs = viz_result_rgbs.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                        
                        # NOTE: 这里初始化了一个list，需要可视化的东西可以后续加上去
                        img_list = [viz_rgbs * 255, viz_result_rgbs * 255]

                        image_diff = np.abs(viz_rgbs.numpy() - viz_result_rgbs.numpy()).mean(2) # .clip(0.2) / 0.2
                        image_diff_color = cv2.applyColorMap((image_diff*255).astype(np.uint8), cv2.COLORMAP_JET)
                        image_diff_color = cv2.cvtColor(image_diff_color, cv2.COLOR_RGB2BGR)
                        img_list.append(torch.from_numpy(image_diff_color))
                        
                        
                        prepare_depth_normal_visual(img_list, self.hparams, metadata_item, typ, results, Runner.visualize_scalars, experiment_path_current, i)
                        
                        get_semantic_gt_pred(results, val_type, metadata_item, viz_rgbs, self.logits_2_label, typ, remapping, img_list, experiment_path_current, 
                                            i, self.writer, self.hparams, viz_result_rgbs * 255, self.metrics_val, self.metrics_val_each)
                        
                        
                        if self.hparams.enable_instance:
                            instances, p_instances, all_points_rgb, all_points_semantics, gt_points_semantic, p_instances_building = get_instance_pred(
                                results, val_type, metadata_item, viz_rgbs, self.logits_2_label, typ, remapping, 
                                experiment_path_current, i, self.writer, self.hparams, viz_result_rgbs, self.thing_classes,
                                all_points_rgb, all_points_semantics, gt_points_semantic)
                            
                            all_instance_features.append(instances)
                            all_thing_features.append(p_instances)
                            if not self.hparams.only_train_building:
                                all_thing_features_building.append(p_instances_building)

                            gt_points_rgb.append(viz_rgbs.view(-1,3))



                        # NOTE: 对需要可视化的list进行处理
                        # save images: list：  N * (H, W, 3),  -> tensor(N, 3, H, W)
                        # 将None元素转换为zeros矩阵
                        img_list = [torch.zeros_like(viz_rgbs) if element is None else element for element in img_list]
                        img_list = torch.stack(img_list).permute(0,3,1,2)
                        img = make_grid(img_list, nrow=3)
                        img_grid = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        Image.fromarray(img_grid).save(str(experiment_path_current / 'val_rgbs' / ("%06d_all.jpg" % i)))

                        if self.writer is not None and (train_index % 50000 == 0):
                            self.writer.add_image('5_val_images/{}'.format(i), img.byte(), train_index)
                        if self.wandb is not None and (train_index % 50000 == 0):
                            Img = wandb.Image(img, caption="ckpt {}: {} th".format(train_index, i))
                            self.wandb.log({"images_all/{}".format(train_index): Img, 'epoch': i})
                        

                        if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'pred_rgb')):
                            Path(str(experiment_path_current / 'val_rgbs' / 'pred_rgb')).mkdir()
                        Image.fromarray((viz_result_rgbs.numpy() * 255).astype(np.uint8)).save(
                            str(experiment_path_current / 'val_rgbs' / 'pred_rgb' / ("%06d_pred_rgb.jpg" % i)))
                        
                        
                        if val_type == 'val':
                            #save  [pred_label, pred_rgb, fg_bg] to the folder 

                            if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'gt_rgb')) and self.hparams.save_individual:
                                Path(str(experiment_path_current / 'val_rgbs' / 'gt_rgb')).mkdir()
                            if self.hparams.save_individual:
                                Image.fromarray((viz_rgbs.numpy() * 255).astype(np.uint8)).save(
                                    str(experiment_path_current / 'val_rgbs' / 'gt_rgb' / ("%06d_gt_rgb.jpg" % i)))


                            if self.hparams.bg_nerf or f'bg_rgb_{typ}' in results:
                                img = Runner._create_fg_bg_image(results[f'fg_rgb_{typ}'].view(viz_rgbs.shape[0],viz_rgbs.shape[1], 3).cpu(),
                                                                results[f'bg_rgb_{typ}'].view(viz_rgbs.shape[0],viz_rgbs.shape[1], 3).cpu())
                                
                                if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'fg_bg')):
                                    Path(str(experiment_path_current / 'val_rgbs' / 'fg_bg')).mkdir()
                                
                                img.save(str(experiment_path_current / 'val_rgbs' / 'fg_bg' / ("%06d_fg_bg.jpg" % i)))
                            
                            # logger
                            samantic_each_value = save_semantic_metric(self.metrics_val_each, CLASSES, samantic_each_value, self.wandb, self.writer, train_index, i)
                            self.metrics_val_each.reset()
                        del results
                
                if self.hparams.enable_instance:
                    
                    # 这里先加一些训练图像
                    if self.hparams.add_train_image and self.hparams.instance_loss_mode != 'linear_assignment':
                        indices_to_train = np.arange(len(self.train_items))
                        indices_to_train = indices_to_train[::20]

                        # indices_to_train = indices_to_train[:1]

                        for k in main_tqdm(indices_to_train):
                            metadata_item = self.train_items[k]
                            results, _ = self.render_image(metadata_item, train_index)
                            typ = 'fine' if 'rgb_fine' in results else 'coarse'

                            if f'instance_map_{typ}' in results:
                                instances = results[f'instance_map_{typ}']
                                device = instances.device

                                # 如果pred semantic存在，则使用
                                # 若不存在， 则创建一个全是things的semantic
                                if f'sem_map_{typ}' in results:
                                    sem_logits = results[f'sem_map_{typ}']
                                    sem_label = self.logits_2_label(sem_logits)
                                    sem_label = remapping(sem_label)
                                else:
                                    sem_label = torch.ones_like(instances)
                                
                                if self.hparams.instance_loss_mode == 'slow_fast':
                                    slow_features = instances[...,self.hparams.num_instance_classes:] 
                                    # all_slow_features.append(slow_features)
                                    instances = instances[...,0:self.hparams.num_instance_classes] # keep fast features only
                                if not self.hparams.render_zyq:
                                    p_instances = create_instances_from_semantics(instances, sem_label, self.thing_classes,device=device)
                                all_thing_features.append(p_instances)





                    # 'linear_assignment' 是直接得到一个伪标签
                    if self.hparams.instance_loss_mode == 'linear_assignment':
                        all_points_instances = torch.stack(all_thing_features, dim=0) # N x d
                        output_dir = str(experiment_path_current / 'panoptic')
                        if not os.path.exists(output_dir):
                            Path(output_dir).mkdir()
                    else:
                            
                        # instance clustering
                        all_instance_features = torch.cat(all_instance_features, dim=0).cpu().numpy()
                        all_thing_features = torch.cat(all_thing_features, dim=0).cpu().numpy() # N x d
                        if not self.hparams.only_train_building:
                            all_thing_features_building = torch.cat(all_thing_features_building, dim=0).cpu().numpy()
                        output_dir = str(experiment_path_current / 'panoptic')
                        if not os.path.exists(output_dir):
                            Path(output_dir).mkdir()
                        if train_index == self.hparams.train_iterations:
                            np.save(os.path.join(output_dir, "all_instance_features.npy"), all_instance_features)
                            np.save(os.path.join(output_dir, "all_points_semantics.npy"), torch.stack(all_points_semantics).cpu().numpy())
                            np.save(os.path.join(output_dir, "all_points_rgb.npy"), torch.stack(all_points_rgb).cpu().numpy())
                            np.save(os.path.join(output_dir, "gt_points_rgb.npy"), torch.stack(gt_points_rgb).cpu().numpy())
                            np.save(os.path.join(output_dir, "gt_points_semantic.npy"), torch.stack(gt_points_semantic).cpu().numpy())
                            np.save(os.path.join(output_dir, "gt_points_instance.npy"), torch.stack(gt_points_instance).cpu().numpy())

                            
                        if self.hparams.cached_centroids_type == 'all':
                            all_points_instances = assign_clusters(all_thing_features, all_points_semantics, all_centroids, 
                                                                    device=self.device, num_images=len(indices_to_eval))
                        elif self.hparams.cached_centroids_type == 'test':
                            if all_centroids is not None:
                                
                                all_points_instances, _ = cluster(all_thing_features, bandwidth=0.2, device=self.device, 
                                                            num_images=len(indices_to_eval), use_dbscan=self.hparams.use_dbscan, all_centroids=all_centroids)
            
                            else:
                                if self.hparams.add_train_image:
                                    all_points_instances, all_centroids = cluster(all_thing_features, bandwidth=0.2, device=self.device, 
                                                                num_images=len(indices_to_eval)+len(indices_to_train), use_dbscan=self.hparams.use_dbscan)
                                else:
                                    all_points_instances, all_centroids = cluster(all_thing_features, bandwidth=0.2, device=self.device, 
                                                                num_images=len(indices_to_eval), use_dbscan=self.hparams.use_dbscan)
                                output_dir = str(experiment_path_current)
                                if not os.path.exists(output_dir):
                                    Path(output_dir).mkdir(parents=True)
                                    
                                all_centroids_path = os.path.join(output_dir, f"test_centroids.npy")
                                with open(all_centroids_path, "wb") as file:
                                    pickle.dump(all_centroids, file)
                                print(f"save all_centroids_cache to : {all_centroids_path}")
                            
                            if not self.hparams.only_train_building:
                                all_points_instances_building, _ = cluster(all_thing_features_building, bandwidth=0.2, device=self.device, 
                                                            num_images=len(indices_to_eval), use_dbscan=self.hparams.use_dbscan)
                                
                    if not os.path.exists(str(experiment_path_current / 'pred_semantics')):
                        Path(str(experiment_path_current / 'pred_semantics')).mkdir()
                    if not os.path.exists(str(experiment_path_current / 'pred_surrogateid')):
                        Path(str(experiment_path_current / 'pred_surrogateid')).mkdir()



                    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                    # import matplotlib.pyplot as plt
                    # lda = LinearDiscriminantAnalysis(n_components=2)
                    # lda_result = lda.fit_transform(all_thing_features[:,1:], all_points_instances.view(-1,all_points_instances.shape[-1]).argmax(dim=1).cpu().numpy())
                    # plt.scatter(lda_result[:, 0], lda_result[:, 1], c=p_instances.argmax(dim=1).cpu().numpy(), cmap='viridis')
                    # plt.xlabel('LDA Component 1')
                    # plt.ylabel('LDA Component 2')
                    # plt.title('LDA Visualization')
                    # # 保存图像为文件（例如，PNG格式）
                    # plt.savefig(os.path.join(str(experiment_path_current / 'panoptic' / ("lda_visualization.jpg"))))
                    # # 关闭图形窗口（如果不需要显示图形界面）
                    # plt.close()


                    for save_i in range(len(indices_to_eval)):
                        p_rgb = all_points_rgb[save_i]
                        p_semantics = all_points_semantics[save_i]
                        # p_semantics = gt_points_semantic[save_i]
                        p_instances = all_points_instances[save_i]
                        if not self.hparams.only_train_building:
                            p_instances_building = all_points_instances_building[save_i]
                        gt_rgb = gt_points_rgb[save_i]
                        gt_semantics = gt_points_semantic[save_i]
                        gt_instances = gt_points_instance[save_i]

                        
                        output_semantics_with_invalid = p_semantics.detach()
                        Image.fromarray(output_semantics_with_invalid.reshape(self.H, self.W).cpu().numpy().astype(np.uint8)).save(
                                str(experiment_path_current / 'pred_semantics'/ ("%06d.png" % int(self.val_items[i].image_path.stem))))
                        if not self.hparams.only_train_building:
                            Image.fromarray(p_instances_building.argmax(dim=1).reshape(self.H, self.W).cpu().numpy().astype(np.uint16)).save(
                                    str(experiment_path_current / 'pred_surrogateid'/ ("%06d.png" % int(self.val_items[i].image_path.stem))))
                        else:
                            Image.fromarray(p_instances.argmax(dim=1).reshape(self.H, self.W).cpu().numpy().astype(np.uint16)).save(
                                    str(experiment_path_current / 'pred_surrogateid'/ ("%06d.png" % int(self.val_items[i].image_path.stem))))
                            
                        
                    # calculate the panoptic quality
                    
                    path_target_sem = os.path.join(self.hparams.dataset_path, 'val', 'labels_gt')
                    path_target_inst = os.path.join(self.hparams.dataset_path, 'val', 'instances_gt')
                    path_pred_sem = str(experiment_path_current / 'pred_semantics')
                    path_pred_inst = str(experiment_path_current / 'pred_surrogateid')
                    if Path(path_target_inst).exists():
                        pq, sq, rq, metrics_each, pred_areas, target_areas, zyq_TP, zyq_FP, zyq_FN, matching = calculate_panoptic_quality_folders(path_pred_sem, path_pred_inst, 
                                        path_target_sem, path_target_inst, image_size=[self.W, self.H])
                        with (experiment_path_current / 'instance.txt').open('w') as f:
                            f.write(f'\n\npred_areas\n')  

                            for key, value in pred_areas.items():
                                f.write(f"    {key}: {value}\n")
                            f.write(f'\n\ntarget_areas\n')  
                            for key, value in target_areas.items():
                                f.write(f"    {key}: {value}\n")
                            
                            f.write(f'\n\nTP\n')  
                            for item in zyq_TP:
                                f.write(    f"{item}\n")
                            
                            f.write(f'\n\nFP\n')  
                            for item in zyq_FP:
                                f.write(    f"{item}\n")
                            
                            f.write(f'\n\nFN\n')  
                            for item in zyq_FN:
                                f.write(f"    {item}\n")

                        val_metrics['pq'] = pq
                        val_metrics['sq'] = sq
                        val_metrics['rq'] = rq
                        val_metrics['metrics_each']=metrics_each
                        if all_centroids is not None:
                            val_metrics['all_centroids_shape'] = all_centroids.shape
                            print(f"all_centroids: {all_centroids.shape}")

                        # 对TP进行处理
                        TP = torch.tensor([value[1] for value in zyq_TP if value[0] == 1])
                        FP = torch.tensor([value[1] for value in zyq_FP if value[0] == 1])
                        FN = torch.tensor([value[1] for value in zyq_FN if value[0] == 1])

                        for save_i in range(len(indices_to_eval)):
                            p_rgb = all_points_rgb[save_i]
                            p_semantics = all_points_semantics[save_i]
                            p_instances = all_points_instances[save_i]
                            if not self.hparams.only_train_building:
                                p_instances_building = all_points_instances_building[save_i]
                            gt_rgb = gt_points_rgb[save_i]
                            gt_semantics = gt_points_semantic[save_i]
                            gt_instances = gt_points_instance[save_i]
                            stack = visualize_panoptic_outputs(
                                p_rgb, p_semantics, p_instances, None, gt_rgb, gt_semantics, gt_instances,
                                self.H, self.W, thing_classes=self.thing_classes, visualize_entropy=False,
                                TP=TP, FP=FP, FN=FN, matching=matching
                            )
                            
                            grid = make_grid(stack, value_range=(0, 1), normalize=True, nrow=4).permute((1, 2, 0)).contiguous()
                            grid = (grid * 255).cpu().numpy().astype(np.uint8)
                            
                            Image.fromarray(grid).save(str(experiment_path_current / 'panoptic' / ("%06d.jpg" % save_i)))
                        

                    ## 对分类结果进行可视化

                    

                # logger
                write_metric_to_folder_logger(self.metrics_val, CLASSES, experiment_path_current, samantic_each_value, self.wandb, self.writer, train_index, self.hparams)
                self.metrics_val.reset()
                
                self.writer.flush()
                self.writer.close()
                self.nerf.train()
            finally:
                if self.is_master and base_tmp_path is not None:
                    shutil.rmtree(base_tmp_path)
            
            return val_metrics, all_centroids
               

    def _run_validation_train(self, train_index=-1, all_centroids=None) -> Dict[str, float]:
        from tools.unetformer.uavid2rgb import remapping
        with torch.inference_mode():
            #semantic 
            self.metrics_val = Evaluator(num_class=self.hparams.num_semantic_classes)
            CLASSES = ('Cluster', 'Building', 'Road', 'Car', 'Tree', 'Vegetation', 'Human', 'Sky', 'Water', 'Ground', 'Mountain')
            self.nerf.eval()
            val_metrics = defaultdict(float)
            base_tmp_path = None
            
            from glob import glob
            used_files = []
            for ext in ('*.png', '*.jpg'):
                used_files.extend(glob(os.path.join(self.hparams.dataset_path, 'subset', 'rgbs', ext)))
            used_files.sort()
            process_item = [Path(far_p).stem for far_p in used_files]


            val_type = self.hparams.val_type  # train  val
            assert val_type == 'train'
            print('val_type: ', val_type)
            try:
                if val_type == 'train':
                    indices_to_eval = np.arange(len(self.train_items))
                    
                experiment_path_current = self.experiment_path / "eval_{}".format(train_index)
                
                    
                samantic_each_value = {}
                for class_name in CLASSES:
                    samantic_each_value[f'{class_name}_iou'] = []
                samantic_each_value['mIoU'] = []
                samantic_each_value['FW_IoU'] = []
                samantic_each_value['F1'] = []

                
                if self.hparams.debug:
                    indices_to_eval = indices_to_eval[:2]
                
                for i in main_tqdm(indices_to_eval):
                    self.metrics_val_each = Evaluator(num_class=self.hparams.num_semantic_classes)

                    metadata_item = self.train_items[i]
                    
                    save_left_or_right = 'val_rgbs'
                    if (metadata_item.left_or_right) != None:
                        if metadata_item.left_or_right == 'left':
                            save_left_or_right = 'val_rgbs_left'
                        elif metadata_item.left_or_right == 'right':
                            save_left_or_right = 'val_rgbs_right'
                    # else:
                        # continue


                    
                    Path(str(experiment_path_current)).mkdir(exist_ok=True)
                    Path(str(experiment_path_current / save_left_or_right)).mkdir(exist_ok=True)
                    
                    ### 渲染 train 的时候只渲染 subset 的
                    if Path(metadata_item.image_path).stem not in process_item:
                        continue
                    file_name = int(Path(metadata_item.image_path).stem)


                    results, _ = self.render_image(metadata_item, train_index)


                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    
                    viz_rgbs = metadata_item.load_image().float() / 255.
                    self.H, self.W = viz_rgbs.shape[0], viz_rgbs.shape[1]
                    
                    # get rendering rgbs
                    viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()


                    viz_result_rgbs = viz_result_rgbs.clamp(0,1)
                    viz_result_rgbs = viz_result_rgbs.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                    
                    if not self.hparams.enable_instance:

                        # NOTE: 这里初始化了一个list，需要可视化的东西可以后续加上去
                        img_list = [viz_rgbs * 255, viz_result_rgbs * 255]

                        image_diff = np.abs(viz_rgbs.numpy() - viz_result_rgbs.numpy()).mean(2) # .clip(0.2) / 0.2
                        image_diff_color = cv2.applyColorMap((image_diff*255).astype(np.uint8), cv2.COLORMAP_JET)
                        image_diff_color = cv2.cvtColor(image_diff_color, cv2.COLOR_RGB2BGR)
                        img_list.append(torch.from_numpy(image_diff_color))
                        
                        prepare_depth_normal_visual(img_list, self.hparams, metadata_item, typ, results, Runner.visualize_scalars, experiment_path_current, file_name,save_left_or_right)
                        
                        get_semantic_gt_pred(results, val_type, metadata_item, viz_rgbs, self.logits_2_label, typ, remapping, img_list, experiment_path_current, 
                                            file_name, self.writer, self.hparams, viz_result_rgbs * 255, self.metrics_val, self.metrics_val_each,save_left_or_right)


                        if not os.path.exists(str(experiment_path_current / save_left_or_right / 'pred_rgb')):
                            Path(str(experiment_path_current / save_left_or_right / 'pred_rgb')).mkdir()
                        Image.fromarray((viz_result_rgbs.numpy() * 255).astype(np.uint8)).save(
                            str(experiment_path_current / save_left_or_right / 'pred_rgb' / ("%06d_pred_rgb.jpg" % i)))
                        

                        # NOTE: 对需要可视化的list进行处理
                        # save images: list：  N * (H, W, 3),  -> tensor(N, 3, H, W)
                        # 将None元素转换为zeros矩阵
                        img_list = [torch.zeros_like(viz_rgbs) if element is None else element for element in img_list]
                        img_list = torch.stack(img_list).permute(0,3,1,2)
                        img = make_grid(img_list, nrow=3)
                        img_grid = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                        if not os.path.exists(str(experiment_path_current / save_left_or_right / 'all')):
                            Path(str(experiment_path_current / save_left_or_right / 'all')).mkdir()
                        Image.fromarray(img_grid).save(str(experiment_path_current / save_left_or_right / 'all' / ("%06d_all.jpg" % file_name)))


                    if self.hparams.enable_instance:  #  这里对每一张instance进行聚类
                                
                        all_instance_features, all_thing_features = [], []
                        all_thing_features_building = []
                        all_points_rgb, all_points_semantics = [], []

                        instances, p_instances, all_points_rgb, all_points_semantics, gt_points_semantic, p_instances_building = get_instance_pred(
                            results, val_type, metadata_item, viz_rgbs, self.logits_2_label, typ, remapping, 
                            experiment_path_current, file_name, self.writer, self.hparams, viz_result_rgbs, self.thing_classes,
                            all_points_rgb, all_points_semantics)
                        
                        all_instance_features.append(instances)
                        all_thing_features.append(p_instances)
                        
                        # 'linear_assignment' 是直接得到一个伪标签
                        if self.hparams.instance_loss_mode == 'linear_assignment':
                            all_points_instances = torch.stack(all_thing_features, dim=0) # N x d
                            output_dir = str(experiment_path_current / 'panoptic')
                            if not os.path.exists(output_dir):
                                Path(output_dir).mkdir()
                        else:
                            # instance clustering
                            all_instance_features = torch.cat(all_instance_features, dim=0).cpu().numpy()
                            all_thing_features = torch.cat(all_thing_features, dim=0).cpu().numpy() # N x d
                            
                            output_dir = str(experiment_path_current / 'panoptic')
                            if not os.path.exists(output_dir):
                                Path(output_dir).mkdir()

                            assert all_centroids is not None
                            # # 这里送入的是单张图片
                            # all_points_instances, _ = cluster(all_thing_features, bandwidth=0.2, device=self.device, 
                            #                             num_images=1, use_dbscan=self.hparams.use_dbscan, all_centroids=all_centroids)
                            all_points_instances = assign_clusters(all_thing_features, all_points_semantics, all_centroids, 
                                                                device=self.device, num_images=1)
                                
                        if not os.path.exists(str(experiment_path_current / 'pred_semantics')):
                            Path(str(experiment_path_current / 'pred_semantics')).mkdir()
                        if not os.path.exists(str(experiment_path_current / 'pred_surrogateid')):
                            Path(str(experiment_path_current / 'pred_surrogateid')).mkdir()


                        for save_i in range(1):
                            p_rgb = all_points_rgb[save_i]
                            p_semantics = all_points_semantics[save_i]
                            p_instances = all_points_instances[save_i]

                            
                            output_semantics_with_invalid = p_semantics.detach()
                            Image.fromarray(output_semantics_with_invalid.reshape(self.H, self.W).cpu().numpy().astype(np.uint8)).save(
                                    str(experiment_path_current / 'pred_semantics'/ ("%06d.png" % metadata_item.image_index)))
                        
                            Image.fromarray(p_instances.argmax(dim=1).reshape(self.H, self.W).cpu().numpy().astype(np.uint16)).save(
                                    str(experiment_path_current / 'pred_surrogateid'/ ("%06d.png" % metadata_item.image_index)))
                                
            
                            stack = visualize_panoptic_outputs(
                                p_rgb, p_semantics, p_instances, None, None, None, None,
                                self.H, self.W, thing_classes=self.thing_classes, visualize_entropy=False,
                            )
                            
                            grid = make_grid(stack, value_range=(0, 1), normalize=True, nrow=4).permute((1, 2, 0)).contiguous()
                            grid = (grid * 255).cpu().numpy().astype(np.uint8)
                            
                            Image.fromarray(grid).save(str(experiment_path_current / 'panoptic' / ("%06d.jpg" % metadata_item.image_index)))
                                
                    del results

                    ## 对分类结果进行可视化
                self.writer.flush()
                self.writer.close()
                self.nerf.train()
            finally:
                if self.is_master and base_tmp_path is not None:
                    shutil.rmtree(base_tmp_path)
            
            return val_metrics, all_centroids
               

    def _run_validation_instance(self, train_index=-1, all_centroids=None) -> Dict[str, float]:
        from tools.unetformer.uavid2rgb import remapping
        with torch.inference_mode():
            #semantic 
            self.metrics_val = Evaluator(num_class=self.hparams.num_semantic_classes)
            CLASSES = ('Cluster', 'Building', 'Road', 'Car', 'Tree', 'Vegetation', 'Human', 'Sky', 'Water', 'Ground', 'Mountain')
            self.nerf.eval()
            val_metrics = defaultdict(float)
            base_tmp_path = None
            

            val_paths = sorted(list((Path(self.hparams.dataset_path) / self.hparams.supp_name / 'metadata').iterdir()))
            # val_paths = sorted(list((Path(self.hparams.dataset_path) / 'val' / 'metadata').iterdir()))


            train_paths = val_paths
            train_paths.sort(key=lambda x: x.name)
            val_paths_set = set(val_paths)
            image_indices = {}
            for i, train_path in enumerate(train_paths):
                image_indices[train_path.name] = i
            render_items = [
                self._get_metadata_item(x, image_indices[x.name], self.hparams.val_scale_factor, x in val_paths_set) for x
                in train_paths]

            H = render_items[0].H
            W = render_items[0].W

            if self.hparams.start == -1 and self.hparams.end == -1:
                indices_to_eval = np.arange(len(render_items))
            else:
                indices_to_eval = np.arange(len(render_items))[self.hparams.start:self.hparams.end]
            
            # indices_to_eval = indices_to_eval[-1:]

            val_type = self.hparams.val_type  # train  val
            print('val_type: ', val_type)

            try:                    
                experiment_path_current = self.experiment_path / "eval_{}".format(train_index)
                
                    
                samantic_each_value = {}
                for class_name in CLASSES:
                    samantic_each_value[f'{class_name}_iou'] = []
                samantic_each_value['mIoU'] = []
                samantic_each_value['FW_IoU'] = []
                samantic_each_value['F1'] = []

                
                for i in main_tqdm(indices_to_eval):
                    self.metrics_val_each = Evaluator(num_class=self.hparams.num_semantic_classes)
                    
                    metadata_item = render_items[i]
                    i = int(Path(metadata_item.depth_dji_path).stem)
                    file_name = int(Path(metadata_item.depth_dji_path).stem)
                    self.hparams.sampling_mesh_guidance = False

                    save_left_or_right = 'val_rgbs'

                    Path(str(experiment_path_current)).mkdir(exist_ok=True)
                    Path(str(experiment_path_current / save_left_or_right)).mkdir(exist_ok=True)
                    

                    results, _ = self.render_image(metadata_item, train_index)


                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    
                    viz_rgbs = torch.zeros((H,W,3)).to(self.device).cpu()
                    self.H, self.W = H, W
                    
                    # get rendering rgbs
                    viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()


                    viz_result_rgbs = viz_result_rgbs.clamp(0,1)
                    viz_result_rgbs = viz_result_rgbs.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                    
                

                    # NOTE: 这里初始化了一个list，需要可视化的东西可以后续加上去
                    img_list = [viz_result_rgbs * 255]


                    prepare_depth_normal_visual(img_list, self.hparams, metadata_item, typ, results, Runner.visualize_scalars, experiment_path_current, file_name,save_left_or_right, self.ray_altitude_range)
                    

                    # get_semantic_gt_pred_render_zyq(results, 'val', metadata_item, viz_result_rgbs, self.logits_2_label, typ, remapping,
                    #                     self.metrics_val, self.metrics_val_each, img_list, experiment_path_current, i, self.writer, self.hparams)
                    
                    get_semantic_gt_pred(results, val_type, metadata_item, viz_rgbs, self.logits_2_label, typ, remapping, img_list, experiment_path_current, 
                                        file_name, self.writer, self.hparams, viz_result_rgbs * 255, self.metrics_val, self.metrics_val_each,save_left_or_right)


                    if not os.path.exists(str(experiment_path_current / save_left_or_right / 'pred_rgb')):
                        Path(str(experiment_path_current / save_left_or_right / 'pred_rgb')).mkdir()
                    Image.fromarray((viz_result_rgbs.numpy() * 255).astype(np.uint8)).save(
                        str(experiment_path_current / save_left_or_right / 'pred_rgb' / ("%06d_pred_rgb.jpg" % i)))
                    

                    # NOTE: 对需要可视化的list进行处理
                    # save images: list：  N * (H, W, 3),  -> tensor(N, 3, H, W)
                    # 将None元素转换为zeros矩阵
                    img_list = [torch.zeros_like(viz_rgbs) if element is None else element for element in img_list]
                    img_list = torch.stack(img_list).permute(0,3,1,2)
                    img = make_grid(img_list, nrow=3)
                    img_grid = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                    if not os.path.exists(str(experiment_path_current / save_left_or_right / 'all')):
                        Path(str(experiment_path_current / save_left_or_right / 'all')).mkdir()
                    Image.fromarray(img_grid).save(str(experiment_path_current / save_left_or_right / 'all' / ("%06d_all.jpg" % file_name)))


                    if self.hparams.enable_instance:  #  这里对每一张instance进行聚类
                                
                        all_instance_features, all_thing_features = [], []
                        all_thing_features_building = []
                        all_points_rgb, all_points_semantics = [], []

                        instances, p_instances, all_points_rgb, all_points_semantics, gt_points_semantic, p_instances_building = get_instance_pred(
                            results, val_type, metadata_item, viz_rgbs, self.logits_2_label, typ, remapping, 
                            experiment_path_current, file_name, self.writer, self.hparams, viz_result_rgbs, self.thing_classes,
                            all_points_rgb, all_points_semantics)
                        
                        all_instance_features.append(instances)
                        all_thing_features.append(p_instances)
                        
                        # 'linear_assignment' 是直接得到一个伪标签
                        if self.hparams.instance_loss_mode == 'linear_assignment':
                            all_points_instances = torch.stack(all_thing_features, dim=0) # N x d
                            output_dir = str(experiment_path_current / 'panoptic')
                            if not os.path.exists(output_dir):
                                Path(output_dir).mkdir()
                        else:
                            # instance clustering
                            all_instance_features = torch.cat(all_instance_features, dim=0).cpu().numpy()
                            all_thing_features = torch.cat(all_thing_features, dim=0).cpu().numpy() # N x d
                            
                            output_dir = str(experiment_path_current / 'panoptic')
                            if not os.path.exists(output_dir):
                                Path(output_dir).mkdir()

                            assert all_centroids is not None
                            # # 这里送入的是单张图片
                            # all_points_instances, _ = cluster(all_thing_features, bandwidth=0.2, device=self.device, 
                            #                             num_images=1, use_dbscan=self.hparams.use_dbscan, all_centroids=all_centroids)
                            all_points_instances = assign_clusters(all_thing_features, all_points_semantics, all_centroids, 
                                                                device=self.device, num_images=1)
                                
                        if not os.path.exists(str(experiment_path_current / 'pred_semantics')):
                            Path(str(experiment_path_current / 'pred_semantics')).mkdir()
                        if not os.path.exists(str(experiment_path_current / 'pred_surrogateid')):
                            Path(str(experiment_path_current / 'pred_surrogateid')).mkdir()


                        for save_i in range(1):
                            p_rgb = all_points_rgb[save_i]
                            p_semantics = all_points_semantics[save_i]
                            p_instances = all_points_instances[save_i]

                            
                            output_semantics_with_invalid = p_semantics.detach()
                            Image.fromarray(output_semantics_with_invalid.reshape(self.H, self.W).cpu().numpy().astype(np.uint8)).save(
                                    str(experiment_path_current / 'pred_semantics'/ ("%06d.png" % metadata_item.image_index)))
                        
                            Image.fromarray(p_instances.argmax(dim=1).reshape(self.H, self.W).cpu().numpy().astype(np.uint16)).save(
                                    str(experiment_path_current / 'pred_surrogateid'/ ("%06d.png" % metadata_item.image_index)))
                                
            
                            stack = visualize_panoptic_outputs(
                                p_rgb, p_semantics, p_instances, None, None, None, None,
                                self.H, self.W, thing_classes=self.thing_classes, visualize_entropy=False,
                            )
                            
                            grid = make_grid(stack, value_range=(0, 1), normalize=True, nrow=4).permute((1, 2, 0)).contiguous()
                            grid = (grid * 255).cpu().numpy().astype(np.uint8)
                            
                            Image.fromarray(grid).save(str(experiment_path_current / 'panoptic' / ("%06d.jpg" % metadata_item.image_index)))
                                
                    del results

                    ## 对分类结果进行可视化
                self.writer.flush()
                self.writer.close()
                self.nerf.train()
            finally:
                if self.is_master and base_tmp_path is not None:
                    shutil.rmtree(base_tmp_path)
            
            return val_metrics, all_centroids
               


    def val_3d_to_2d(self):
        self._setup_experiment_dir()
        val_metrics = self._run_validation_val_3d_to_2d(0)
        self._write_final_metrics(val_metrics, train_iterations=0)

    def _run_validation_val_3d_to_2d(self, train_index=-1) -> Dict[str, float]:

        from tools.unetformer.uavid2rgb import remapping
        with torch.inference_mode():
            #semantic 
            self.metrics_val = Evaluator(num_class=self.hparams.num_semantic_classes)
            CLASSES = ('Cluster', 'Building', 'Road', 'Car', 'Tree', 'Vegetation', 'Human', 'Sky', 'Water', 'Ground', 'Mountain')
            self.nerf.eval()
            val_metrics = defaultdict(float)
            base_tmp_path = None
            
            val_type = 'val'  # train  val
            print('val_type: ', val_type)
 
            indices_to_eval = np.arange(len(self.val_items))
            
            experiment_path_current = self.experiment_path / "eval_{}".format(train_index)
            Path(str(experiment_path_current)).mkdir()
            Path(str(experiment_path_current / 'val_rgbs')).mkdir()
            with (experiment_path_current / 'psnr.txt').open('w') as f:
                
                samantic_each_value = {}
                for class_name in CLASSES:
                    samantic_each_value[f'{class_name}_iou'] = []
                samantic_each_value['mIoU'] = []
                samantic_each_value['FW_IoU'] = []
                samantic_each_value['F1'] = []

                
                for i in main_tqdm(indices_to_eval):
                    self.metrics_val_each = Evaluator(num_class=self.hparams.num_semantic_classes)
                    metadata_item = self.val_items[i]
                    
                    typ = 'fine'
                    
                    viz_rgbs = metadata_item.load_image().float() / 255.
                    
                    
                    # NOTE: 这里初始化了一个list，需要可视化的东西可以后续加上去
                    img_list = [viz_rgbs * 255]

                    gt_label = metadata_item.load_gt()
                    gt_label = remapping(gt_label)
                    
                    # 这里读取点云投影的图片，或者进行点云投影
                    # sem_label

                    sem_label_path = os.path.join(self.hparams.dataset_path, 'val', self.hparams.label_name_3d_to_2d,f"{metadata_item.label_path.stem}.png")
                    sem_label = Image.open(sem_label_path)    #.convert('RGB')
                    sem_label =  torch.ByteTensor(np.asarray(sem_label)).view(-1)
                    
                    sem_label = remapping(sem_label)
                    
                    gt_label_rgb = custom2rgb(gt_label.view(*viz_rgbs.shape[:-1]).cpu().numpy())
                    visualize_sem = custom2rgb(sem_label.view(*viz_rgbs.shape[:-1]).cpu().numpy())
                    if self.hparams.remove_cluster:
                        gt_label = gt_label.view(-1)
                        sem_label = sem_label.view(-1)
                        gt_no_zero_mask = (gt_label != 0)
                        # pred_no_zero_mask = (sem_label != 0)
                        # no_cluster_mask = gt_no_zero_mask * pred_no_zero_mask
                        gt_label_ig = gt_label[gt_no_zero_mask]
                        sem_label_ig = sem_label[gt_no_zero_mask]
                        self.metrics_val.add_batch(gt_label_ig.cpu().numpy(), sem_label_ig.cpu().numpy())
                        self.metrics_val_each.add_batch(gt_label_ig.view(-1).cpu().numpy(), sem_label_ig.cpu().numpy())
                    else:
                        self.metrics_val.add_batch(gt_label.view(-1).cpu().numpy(), sem_label.cpu().numpy())
                        self.metrics_val_each.add_batch(gt_label.view(-1).cpu().numpy(), sem_label.cpu().numpy())
                    

                    img_list.append(torch.from_numpy(gt_label_rgb))
                    img_list.append(torch.from_numpy(visualize_sem))
                    
                    if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'label_pc')):
                        Path(str(experiment_path_current / 'val_rgbs' / 'label_pc')).mkdir()
                    Image.fromarray((visualize_sem).astype(np.uint8)).save(str(experiment_path_current / 'val_rgbs' / 'label_pc' / ("%06d_label_pc.jpg" % i)))

                    if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'gt_label')):
                        Path(str(experiment_path_current / 'val_rgbs' / 'gt_label')).mkdir()
                    
                    Image.fromarray((gt_label_rgb).astype(np.uint8)).save(str(experiment_path_current / 'val_rgbs' / 'gt_label' / ("%06d_gt_label.jpg" % i)))



                    # NOTE: 对需要可视化的list进行处理
                    # save images: list：  N * (H, W, 3),  -> tensor(N, 3, H, W)
                    # 将None元素转换为zeros矩阵
                    img_list = [torch.zeros_like(viz_rgbs) if element is None else element for element in img_list]
                    img_list = torch.stack(img_list).permute(0,3,1,2)
                    img = make_grid(img_list, nrow=3)
                    img_grid = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    Image.fromarray(img_grid).save(str(experiment_path_current / 'val_rgbs' / ("%06d_all.jpg" % i)))

                    
                    if val_type == 'val':
                        samantic_each_value = save_semantic_metric(self.metrics_val_each, CLASSES, samantic_each_value, self.wandb, self.writer, train_index, i)
                        self.metrics_val_each.reset()

            # logger
            write_metric_to_folder_logger(self.metrics_val, CLASSES, experiment_path_current, samantic_each_value, self.wandb, self.writer, train_index, self.hparams)
            self.metrics_val.reset()
            
            self.writer.flush()
            self.writer.close()
            self.nerf.train()

        return val_metrics
    

    def _save_checkpoint(self, optimizers: Dict[str, any], scaler: GradScaler, train_index: int, dataset_index: int,
                         dataset_state: Optional[str]) -> None:
        dict = {
            'model_state_dict': self.nerf.state_dict(),
            'scaler': scaler.state_dict(),
            'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
            'iteration': train_index,
            'torch_random_state': torch.get_rng_state(),
            'np_random_state': np.random.get_state(),
            'random_state': random.getstate(),
            'dataset_index': dataset_index
        }

        if dataset_state is not None:
            dict['dataset_state'] = dataset_state

        if self.bg_nerf is not None:
            dict['bg_model_state_dict'] = self.bg_nerf.state_dict()

        torch.save(dict, self.model_path / '{}.pt'.format(train_index))

    def render_image(self, metadata: ImageMetadata, train_index=-1) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if 'sdf' in self.hparams.network_type:
            if self.hparams.sdf_as_gpnerf:
                from gp_nerf.rendering_gpnerf_clean_sdf_as_gpnerf import render_rays
            else:
                from gp_nerf.rendering_gpnerf_clean_sdf import render_rays
        else:
            from gp_nerf.rendering_gpnerf import render_rays
        directions = get_ray_directions(metadata.W,
                                        metadata.H,
                                        metadata.intrinsics[0],
                                        metadata.intrinsics[1],
                                        metadata.intrinsics[2],
                                        metadata.intrinsics[3],
                                        self.hparams.center_pixels,
                                        self.device)
        depth_scale = torch.abs(directions[:, :, 2]).view(-1)

        with torch.cuda.amp.autocast(enabled=self.hparams.amp):
            ###############3 . 俯视图，  render0.3视角下第一张图片
            if self.hparams.render_zyq and self.hparams.enable_instance and self.hparams.fushi:
                if 'Yingrenshi' in self.hparams.dataset_path:
                    image_rays = get_rays(directions, metadata.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)
                    ray_d = image_rays[int(metadata.H/2), int(metadata.W/2), 3:6]
                    ray_o = image_rays[int(metadata.H/2), int(metadata.W/2), :3]

                    z_vals_inbound = 1
                    new_o = ray_o - ray_d * z_vals_inbound
                    metadata.c2w[:,3]= new_o
                    metadata.c2w[1:3,3]=self.sphere_center[1:3]
                    def rad(x):
                        return math.radians(x)
                    angle=30
                    cosine = math.cos(rad(angle))
                    sine = math.sin(rad(angle))
                    rotation_matrix_x = torch.tensor([[1, 0, 0],
                                    [0, cosine, sine],
                                    [0, -sine, cosine]])
                    angle=-40
                    cosine = math.cos(rad(angle))
                    sine = math.sin(rad(angle))
                    rotation_matrix_y = torch.tensor([[cosine, 0, sine],
                                    [0, 1, 0],
                                    [-sine, 0, cosine]])
                    metadata.c2w[:3,:3]=rotation_matrix_y @ (rotation_matrix_x @ metadata.c2w[:3,:3])
                    metadata.c2w[1,3]=metadata.c2w[1,3]-0.4
                    metadata.c2w[2,3]=metadata.c2w[2,3]+0.05
                elif 'Longhua_block' in self.hparams.dataset_path:

                    image_rays = get_rays(directions, metadata.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)
                    ray_d = image_rays[int(metadata.H/2), int(metadata.W/2), 3:6]
                    ray_o = image_rays[int(metadata.H/2), int(metadata.W/2), :3]

                    z_vals_inbound = 1
                    new_o = ray_o - ray_d * z_vals_inbound
                    metadata.c2w[:,3]= new_o
                    metadata.c2w[1:3,3]=self.sphere_center[1:3]
                elif 'Campus_new' in self.hparams.dataset_path:

                    image_rays = get_rays(directions, metadata.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)
                    ray_d = image_rays[int(metadata.H/2), int(metadata.W/2), 3:6]
                    ray_o = image_rays[int(metadata.H/2), int(metadata.W/2), :3]

                    z_vals_inbound = 0.6
                    new_o = ray_o - ray_d * z_vals_inbound
                    metadata.c2w[:,3]= new_o
                    # metadata.c2w[1:3,3]=self.sphere_center[1:3]
                    def rad(x):
                        return math.radians(x)
                    angle=0 #-10
                    cosine = math.cos(rad(angle))
                    sine = math.sin(rad(angle))
                    rotation_matrix_x = torch.tensor([[1, 0, 0],
                                    [0, cosine, sine],
                                    [0, -sine, cosine]])
                    angle=0
                    cosine = math.cos(rad(angle))
                    sine = math.sin(rad(angle))
                    rotation_matrix_y = torch.tensor([[cosine, 0, sine],
                                    [0, 1, 0],
                                    [-sine, 0, cosine]])
                    metadata.c2w[:3,:3]=rotation_matrix_y @ (rotation_matrix_x @ metadata.c2w[:3,:3])
                    # metadata.c2w[1,3]=metadata.c2w[1,3]-0.4
                    # metadata.c2w[2,3]=metadata.c2w[2,3]+0.05

                    
                    
            #########################################################


            rays = get_rays(directions, metadata.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)

            rays = rays.view(-1, 8).to(self.device, non_blocking=True).cuda()  # (H*W, 8)
            if self.hparams.render_zyq:
                image_indices = metadata.image_index * torch.ones(rays.shape[0], device=rays.device)
                # image_indices = metadata.image_index * torch.ones(rays.shape[0], device=rays.device)
            elif 'train' in self.hparams.val_type:
                if 'b1' in self.hparams.dataset_path:
                    image_indices = 200 * torch.ones(rays.shape[0], device=rays.device)
                else:    
                    image_indices = 300 * torch.ones(rays.shape[0], device=rays.device)

                # image_indices = 0 * torch.ones(rays.shape[0], device=rays.device)
            else:
                image_indices = metadata.image_index * torch.ones(rays.shape[0], device=rays.device) \
                    if self.hparams.appearance_dim > 0 else None
            results = {}



            if 'RANK' in os.environ:
                nerf = self.nerf.module
            else:
                nerf = self.nerf

            if self.bg_nerf is not None and 'RANK' in os.environ:
                bg_nerf = self.bg_nerf.module
            else:
                bg_nerf = self.bg_nerf

            for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
                if self.hparams.depth_dji_type == "mesh" and self.hparams.sampling_mesh_guidance:
                    if 'train' in self.hparams.val_type:
                        # # print('load depth')
                        # gt_depths = metadata.load_depth_dji().view(-1).to(self.device)
                        # gt_depths = gt_depths / depth_scale    # 这里读的是depth mesh， 存储的是z分量
                        gt_depths = None
                        
                    else:
                            
                        gt_depths = metadata.load_depth_dji().view(-1).to(self.device)
                        gt_depths = gt_depths / depth_scale    # 这里读的是depth mesh， 存储的是z分量
                else: 
                    gt_depths = None
                result_batch, _ = render_rays(nerf=nerf, bg_nerf=bg_nerf,
                                              rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                              image_indices=image_indices[
                                                            i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                                              hparams=self.hparams,
                                              sphere_center=self.sphere_center,
                                              sphere_radius=self.sphere_radius,
                                              get_depth=True,
                                              get_depth_variance=False,
                                              get_bg_fg_rgb=True,
                                              train_iterations=train_index,
                                              gt_depths= gt_depths[i:i + self.hparams.image_pixel_batch_size] if gt_depths is not None else None,
                                              pose_scale_factor = self.pose_scale_factor)
                if 'air_sigma_loss' in result_batch:
                    del result_batch['air_sigma_loss']
                for key, value in result_batch.items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value.cpu())

            for key, value in results.items():
                results[key] = torch.cat(value)
            return results, rays

    @staticmethod
    def _create_result_image(rgbs: torch.Tensor, result_rgbs: torch.Tensor, result_depths: torch.Tensor) -> Image:
        if result_depths is not None:
            depth_vis = Runner.visualize_scalars(torch.log(result_depths + 1e-8).view(rgbs.shape[0], rgbs.shape[1]).cpu())
            images = (rgbs * 255, result_rgbs * 255, depth_vis)
            return Image.fromarray(np.concatenate(images, 1).astype(np.uint8))
        else:
            images = (rgbs * 255, result_rgbs * 255) #, depth_vis)
            return Image.fromarray(np.concatenate(images, 1).astype(np.uint8))
    
    def _create_fg_bg_image(fgs: torch.Tensor, bgs: torch.Tensor) -> Image:
        images = (fgs * 255, bgs * 255) #, depth_vis)
        return Image.fromarray(np.concatenate(images, 1).astype(np.uint8))
    

    def _create_rendering_semantic(rgbs: torch.Tensor, gt_semantic: torch.Tensor, pseudo_semantic: torch.Tensor,
                                   pred_rgb: torch.Tensor, pred_semantic: torch.Tensor, pred_depth_or_normal: torch.Tensor) -> Image:
        if gt_semantic is None:
            gt_semantic = torch.zeros_like(rgbs)
        if pred_depth_or_normal is None:
            pred_depth_or_normal = torch.zeros_like(rgbs)
        else:
            pred_depth_or_normal = (pred_depth_or_normal+1)*0.5*255
        image_1 = (rgbs * 255, gt_semantic, pseudo_semantic)
        image_1 = Image.fromarray(np.concatenate(image_1, 1).astype(np.uint8))
        image_2 = (pred_rgb * 255, pred_semantic, pred_depth_or_normal)
        image_2 = Image.fromarray(np.concatenate(image_2, 1).astype(np.uint8))
        
        return Image.fromarray(np.concatenate((image_1, image_2), 0).astype(np.uint8))

    def _create_res_list(rgbs, gt_semantic, pseudo_semantic, pred_rgb, pred_semantic, pred_normal) -> Image:
        if gt_semantic is None:
            gt_semantic = torch.zeros_like(rgbs)
        pred_normal = (pred_normal+1)*0.5*255

        res_list = [rgbs * 255, gt_semantic, pseudo_semantic, pred_rgb * 255, pred_semantic, pred_normal]
        
        return res_list
    
    @staticmethod
    # def visualize_scalars(scalar_tensor: torch.Tensor, invalid_mask=None) -> np.ndarray:
    #     if invalid_mask is not None:
    #         w, h, _ = invalid_mask.shape
    #         to_use = scalar_tensor[~invalid_mask].view(-1)
    #     else:
    #         to_use = scalar_tensor.view(-1)
    #     while to_use.shape[0] > 2 ** 24:
    #         to_use = to_use[::2]

    #     mi = torch.quantile(to_use, 0.05)
    #     ma = torch.quantile(to_use, 0.95)

    #     scalar_tensor = (scalar_tensor - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    #     scalar_tensor = scalar_tensor.clamp_(0, 1)

    #     scalar_tensor = ((1 - scalar_tensor) * 255).byte().numpy()  # inverse heatmap
    #     return cv2.cvtColor(cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    
    def visualize_scalars(scalar_tensor: torch.Tensor, ma=None, mi=None, invalid_mask=None) -> np.ndarray:
        
        if ma is not None and mi is not None:
            pass
        else:
            if invalid_mask is not None:
                w, h, _ = invalid_mask.shape
                to_use = scalar_tensor[~invalid_mask].view(-1)
            else:
                to_use = scalar_tensor.view(-1)
            while to_use.shape[0] > 2 ** 24:
                to_use = to_use[::2]

            mi = torch.quantile(to_use, 0.05)
            ma = torch.quantile(to_use, 0.95)
        # print(mi)
        # print(ma)
        scalar_tensor = (scalar_tensor - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
        scalar_tensor = scalar_tensor.clamp_(0, 1)

        scalar_tensor = ((1 - scalar_tensor) * 255).byte().numpy()  # inverse heatmap
        return cv2.cvtColor(cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

    def _get_image_metadata(self) -> Tuple[List[ImageMetadata], List[ImageMetadata]]:
        dataset_path = Path(self.hparams.dataset_path)

        train_path_candidates = sorted(list((dataset_path / 'train' / 'metadata').iterdir()))
        train_paths = [train_path_candidates[i] for i in
                       range(0, len(train_path_candidates), self.hparams.train_every)]
        

        val_paths = sorted(list((dataset_path / 'val' / 'metadata').iterdir()))

        # train_paths=train_paths[:10]
        # val_paths = val_paths[:4]

        # if self.hparams.dataset_type == 'sam':
        #     # train_paths=train_paths[:10]
        #     # val_paths = val_paths[:2]
        #     image_indices_path = train_paths + val_paths
        #     image_indices_path.sort(key=lambda x: x.name)
        #     val_paths_set = set(val_paths)
        #     image_indices = {}
        #     for i, indices_path in enumerate(image_indices_path):
        #         image_indices[indices_path.name] = i
        # else:
        train_paths += val_paths
        train_paths.sort(key=lambda x: x.name)
        val_paths_set = set(val_paths)
        image_indices = {}
        for i, train_path in enumerate(train_paths):
            image_indices[train_path.name] = i

        train_items = [
            self._get_metadata_item(x, image_indices[x.name], self.hparams.train_scale_factor, x in val_paths_set) for x
            in train_paths]
        val_items = [
            self._get_metadata_item(x, image_indices[x.name], self.hparams.val_scale_factor, True) for x in val_paths]

        return train_items, val_items

    def _get_metadata_item(self, metadata_path: Path, image_index: int, scale_factor: int,
                           is_val: bool) -> ImageMetadata:
        image_path = None
        for extension in ['.jpg', '.JPG', '.png', '.PNG']:
            candidate = metadata_path.parent.parent / 'rgbs' / '{}{}'.format(metadata_path.stem, extension)
            if candidate.exists():
                image_path = candidate
                break

        # assert image_path.exists()

        metadata = torch.load(metadata_path, map_location='cpu')
        intrinsics = metadata['intrinsics'] / scale_factor
        # print(f"{metadata['W']} {metadata['H']} {scale_factor}")
        assert metadata['W'] % scale_factor == 0
        assert metadata['H'] % scale_factor == 0

        dataset_mask = metadata_path.parent.parent.parent / 'masks' / metadata_path.name
        if self.hparams.cluster_mask_path is not None:
            if image_index == 0:
                main_print('Using cluster mask path: {}'.format(self.hparams.cluster_mask_path))
            mask_path = Path(self.hparams.cluster_mask_path) / metadata_path.name
        elif dataset_mask.exists():
            if image_index == 0:
                main_print('Using dataset mask path: {}'.format(dataset_mask.parent))
            mask_path = dataset_mask
        else:
            mask_path = None

        
        label_path = None
        for extension in ['.jpg', '.JPG', '.png', '.PNG']:
            
            candidate = metadata_path.parent.parent / f'labels_{self.hparams.label_name}' / '{}{}'.format(metadata_path.stem, extension)

            if candidate.exists():
                label_path = candidate
                break
        instance_path = None
        if self.hparams.enable_instance:
            for extension in ['.jpg', '.JPG', '.png', '.PNG', '.npy']:
                
                candidate = metadata_path.parent.parent / f'{self.hparams.instance_name}' / '{}{}'.format(metadata_path.stem, extension)

                if candidate.exists():
                    instance_path = candidate
                    break
        if 'memory_depth_dji' in self.hparams.dataset_type:
            if self.hparams.depth_dji_type=='las':
                depth_dji_path = os.path.join(metadata_path.parent.parent, 'depth_dji', '%s.npy' % metadata_path.stem) 
            elif self.hparams.depth_dji_type=='mesh':
                depth_dji_path = os.path.join(metadata_path.parent.parent, 'depth_mesh', '%s.npy' % metadata_path.stem) 
            
            if not Path(depth_dji_path).exists() and not self.hparams.render_zyq:
                depth_dji_path=None
            if 'left_or_right' in metadata:
                left_or_right = metadata['left_or_right']
            else:
                left_or_right = None
            return ImageMetadata(image_path, metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                                 intrinsics, image_index, None if (is_val and self.hparams.all_val) else mask_path, is_val, label_path, 
                                 depth_dji_path=depth_dji_path, left_or_right=left_or_right, hparams=self.hparams, instance_path=instance_path)

        else:
            return ImageMetadata(image_path, metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                                intrinsics, image_index, None if (is_val and self.hparams.all_val) else mask_path, is_val, label_path, instance_path=instance_path)

    def _get_experiment_path(self) -> Path:
        exp_dir = Path(self.hparams.exp_name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        existing_versions = [int(x.name) for x in exp_dir.iterdir()]
        version = 0 if len(existing_versions) == 0 else max(existing_versions) + 1
        experiment_path = exp_dir / str(version)
        return experiment_path


    def pretrain_sdf_road_surface_2d(
        self, 
        num_iters=5000, num_points=5000, 
        lr=1.0e-4, w_eikonal=1.0e-3, 
        safe_mse = True, clip_grad_val: float = 0.1, optim_cfg = {}, 
        # Shape configs
        # # e.g. For waymo, +z points to sky, and ego_car is about 0.5m. Hence, floor_dim='z', floor_up_sign=1, ego_height=0.5
        floor_dim: Literal['x','y','z'] = 'x', # The vertical dimension of obj coords
        floor_up_sign: Literal[1, -1]=-1, # [-1] if (-)dim points to sky else [1]
        ego_height: float = 0., # Estimated ego's height from road, in obj coords space
        # Debug & logging related
        logger: Logger=None, log_prefix: str=None, debug_param_detail=False):



        # mesh = trimesh.load_mesh(self.hparams.mesh_path)
        # vertex_normals = mesh.vertex_normals
        # pts_3d = mesh.vertices[vertex_normals[:, 0] <= 0]
        # upper_surface_normal = vertex_normals[vertex_normals[:, 0] <= 0]
        # min_altitude = pts_3d.min(0)[0]
        # min_altitude = (min_altitude - self.hparams.stretch[0][0]) / (self.hparams.stretch[1][0] - self.hparams.stretch[0][0])   

        num_point_each_img = 128
        pts_3d = []
        for idx in range(len(self.train_items)):
            metadata_item = self.train_items[idx]
            depth_map = metadata_item.load_depth_dji()

            H, W = depth_map.shape[:2]
            directions = get_ray_directions(W,
                                        H,
                                        metadata_item.intrinsics[0],
                                        metadata_item.intrinsics[1],
                                        metadata_item.intrinsics[2],
                                        metadata_item.intrinsics[3],
                                        self.hparams.center_pixels,
                                        'cpu')
            depth_map = depth_map.numpy()
            K1 = metadata_item.intrinsics
            K1 = np.array([[K1[0], 0, K1[2]],[0, K1[1], K1[3]],[0,0,1]])
            E1 = np.array(metadata_item.c2w)
            E1 = np.stack([E1[:, 0], E1[:, 1]*-1, E1[:, 2]*-1, E1[:, 3]], 1)

            for p in range(num_point_each_img):
                x, y = random.randint(0, W-1), random.randint(0, H-1)
                depth = depth_map[y, x]   #* 0.5
                if torch.isinf(torch.from_numpy(depth)):
                    continue
                pt_3d = np.dot(np.linalg.inv(K1), np.array([x, y, 1])) * depth
                pt_3d = np.append(pt_3d, 1)
                world_point = np.dot(np.concatenate((E1, [[0,0,0,1]]), 0), pt_3d)
                pts_3d.append(world_point[:3])

        pts_3d = torch.from_numpy(np.array(pts_3d)).to(self.device)
        # pts_3d[:,0:1] = pts_3d.mean(0)[0].expand_as(pts_3d[:,0:1])
        # pts_3d[:,0:1] = pts_3d.min(0).values[0].expand_as(pts_3d[:,0:1])

        

        # visualize_points_list = [pts_3d.view(-1, 3).cpu().numpy()]
        # visualize_points(visualize_points_list)
        tracks_in_obj = contract_to_unisphere_new(pts_3d, self.hparams)
        """
        Pretrain sdf to be a road surface
        """
        floor_dim: int = ['x','y','z'].index(floor_dim)
        other_dims = [i for i in range(3) if i != floor_dim]
        
        device = self.device
        sdf_parameters = list(self.nerf.encoding.parameters())  \
                       + list(self.nerf.plane_encoder.parameters())  \
                       + list(self.nerf.decoder.parameters())
        optimizer = Adam(sdf_parameters, lr=lr, **optim_cfg)
        scaler = GradScaler(init_scale=128.0)
        
        # implicit_surface.preprocess_per_train_step(0)
        self.nerf.encoding.set_anneal_iter(0)
        
        if safe_mse:
            loss_eikonal_fn = lambda x: safe_mse_loss(x, x.new_ones(x.shape), reduction='mean', limit=1.0)
        else:
            loss_eikonal_fn = lambda x: F.mse_loss(x, x.new_ones(x.shape), reduction='mean')
        
        # tracks_in_net = implicit_surface.space.normalize_coords(tracks_in_obj)
        
        # if log_prefix is None: 
        #     log_prefix = implicit_surface.__class__.__name__
        
        with torch.enable_grad():
            with tqdm(range(num_iters), desc=f"=> Pretraining SDF...") as pbar:
                for it in pbar:
                    samples_in_net = torch.empty([num_points,3], dtype=torch.float, device=device).uniform_(-1, 1)
                    # samples_in_obj = self.nerf.encoding.space.unnormalize_coords(samples_in_net)
                    
                    # For each sample point, find the track point of the minimum distance (measured in 3D space.)
                    # # [num_points, 1, 3] - [1, num_tracks, 3] = [num_points, num_tracks, 3] -> [num_points, num_tracks] -> [num_samples]
                    # ret_min_dis_in_obj = (samples_in_obj.unsqueeze(-2) - tracks_in_obj.unsqueeze(0)).norm(dim=-1).min(dim=-1)
                    
                    # For each sample point, find the track point of the minimum distance (measure at 2D space. i.e. xoy plane for floor_dim=z)
                    # [num_points, 1, 3] - [1, num_tracks, 3] = [num_points, num_tracks, 3] -> [num_points, num_tracks] -> [num_samples]
                    ret_min_dis_in_obj = (samples_in_net[..., None, other_dims] - tracks_in_obj[None, ..., other_dims]).norm(dim=-1).min(dim=-1)

                    # For each sample point, current floor'z coordinate of floor_dim
                    floor_at_in_obj = tracks_in_obj[ret_min_dis_in_obj.indices][..., floor_dim] - floor_up_sign * ego_height
                    sdf_gt_in_obj = floor_up_sign * (samples_in_net[..., floor_dim] - floor_at_in_obj)
                    # sign_adjust = torch.where(samples_in_net[:,0] < min_altitude, True, False)
                    # sdf_gt_in_obj[sign_adjust] = torch.abs(sdf_gt_in_obj[sign_adjust])

                    #---- Convert SDF GT value in real-world object's unit to network's unit
                    sdf_gt = sdf_gt_in_obj / self.nerf.sdf_scale
                    # sdf_gt = sdf_gt_in_obj / 25
                    
                    pred_sdf, feature_vector, pred_nablas = self.nerf.forward_sdf_nablas(samples_in_net, nablas_has_grad=True)
                    pred_sdf = pred_sdf
                    nablas_norm = pred_nablas.norm(dim=-1)
                    loss = F.smooth_l1_loss(pred_sdf, sdf_gt, reduction='mean') + w_eikonal * loss_eikonal_fn(nablas_norm)
                    
                    self.writer.add_scalar('sdf_initial/sdf_loss', F.smooth_l1_loss(pred_sdf, sdf_gt, reduction='mean'), it)
                    self.writer.add_scalar('sdf_initial/gradient_error', loss_eikonal_fn(nablas_norm), it)

                    optimizer.zero_grad()
                    
                    # loss.backward()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if clip_grad_val > 0:
                        torch.nn.utils.clip_grad.clip_grad_value_(sdf_parameters, clip_grad_val)
                    
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # pbar.set_postfix(loss=loss.item())
                    # if logger is not None:
                    #     logger.add(f"initialize", log_prefix + '.loss', loss.item(), it)
                    #     logger.add_nested_dict("initialize", log_prefix + '.sdf', tensor_statistics(pred_sdf), it)
                    #     logger.add_nested_dict("initialize", log_prefix + '.nablas_norm', tensor_statistics(nablas_norm), it)
                    #     if debug_param_detail:
                    #         logger.add_nested_dict("initialize", log_prefix + '.encoding', implicit_surface.encoding.stat_param(with_grad=True), it)
    
    def visualize_sdf_surface(self, nerf, save_path, resolution, threshold, device=None):
        print(f"==> Saving mesh to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    sdf, feature_vector, gradient= nerf.forward_sdf_nablas(pts.to(device))

                    # sdf_nn_output= self.nerf.forward_sdf(pts.to(device))
                    # sdf = sdf_nn_output[:, :1]
            return sdf
        
        aabb = self.hparams.stretch
        aabb_min, aabb_max = aabb[0], aabb[1]

        bound=1
        bound_min = torch.FloatTensor([-bound] * 3)
        bound_max = torch.FloatTensor([bound] * 3)

        # bounds_min = torch.FloatTensor([0.2, -1, -1])
        # bounds_max = torch.FloatTensor([1, 1, 1])

        
        vertices, triangles, u = extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, query_func = query_func, use_sdf = True)
        self.save_geometry(vertices, triangles, aabb_max, aabb_min, save_path[:-4]+f'_{threshold}.obj')

        # threshold = u.max()
        # # 为点云创建坐标网格
        # x, y, z = np.meshgrid(np.arange(255), np.arange(255), np.arange(255))
        # x = x.flatten()
        # y = y.flatten()
        # z = z.flatten()
        # vmin = u.min()
        # vmax = u.max()
        # # 将SDF值和坐标连接成点云
        # points = np.vstack((x, y, z)).T

        # b_max_np = bound_max.detach().cpu().numpy()
        # b_min_np = bound_min.detach().cpu().numpy()
        # points = points / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        # points = (points+1)/2
        # points = points * (aabb_max.cpu().numpy() - aabb_min.cpu().numpy()) + aabb_min.cpu().numpy()

        # cmap = plt.get_cmap('viridis')
        # colors = cmap((u - vmin) / (vmax - vmin))[..., :3]  # 将SDF值映射到颜色

        # # 创建TriangleMesh对象
        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(points)
        # mesh.vertex_colors = o3d.utility.Vector3dVector((colors*255).reshape(-1,3))

        # # 保存点云到PLY文件
        # o3d.io.write_triangle_mesh("point_cloud.ply", mesh)

    def save_geometry(self, vertices, triangles, aabb_max, aabb_min, save_path):
        vertices = (vertices+1)/2
        vertices = vertices * (aabb_max.cpu().numpy() - aabb_min.cpu().numpy()) + aabb_min.cpu().numpy()
        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        print(f"==> Finished saving mesh.")

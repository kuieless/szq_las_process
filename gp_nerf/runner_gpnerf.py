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
from tools.unetformer.metric import Evaluator

import pandas as pd

from gp_nerf.eval_utils import get_depth_vis, get_semantic_gt_pred, get_sdf_normal_map, get_semantic_gt_pred_render_zyq
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
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree


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


class Runner:
    def __init__(self, hparams: Namespace, set_experiment_path: bool = True):
        faulthandler.register(signal.SIGUSR1)
        print(f"ignore_index: {hparams.ignore_index}")
        if hparams.balance_weight:
            # cluster 1  building  1 road  1 car 5 tree 5 vegetation 5
            balance_weight = torch.FloatTensor([1, 1, 1, 5, 5, 5, 1, 1, 1, 1, 1]).cuda()
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


    def train(self):

        self._setup_experiment_dir()
        scaler = torch.cuda.amp.GradScaler(enabled=self.hparams.amp)

        if self.hparams.enable_semantic and self.hparams.freeze_geo and self.hparams.ckpt_path is not None:
            
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
            if not self.hparams.enable_semantic:
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
        elif self.hparams.dataset_type == 'sam':
            if self.hparams.add_random_rays:
                from gp_nerf.datasets.memory_dataset_sam_random import MemoryDataset_SAM
            else:
                from gp_nerf.datasets.memory_dataset_sam import MemoryDataset_SAM
            dataset = MemoryDataset_SAM(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
        elif self.hparams.dataset_type == 'sam_project':
            from gp_nerf.datasets.memory_dataset_sam_project import MemoryDataset_SAM
            dataset = MemoryDataset_SAM(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
        elif self.hparams.dataset_type == 'file_normal':
            from gp_nerf.datasets.filesystem_dataset_normal import FilesystemDatasetNormal
            dataset = FilesystemDatasetNormal(self.train_items, self.near, self.far, self.ray_altitude_range,
                                        self.hparams.center_pixels, self.device,
                                        [Path(x) for x in sorted(self.hparams.chunk_paths)], self.hparams.num_chunks,
                                        self.hparams.train_scale_factor, self.hparams.disk_flush_size,self.hparams.desired_chunks)
            # Transform world normal to camera normal
            w2c_rots = torch.linalg.inv(dataset._c2ws[:, :3, :3]).to(self.device, non_blocking=True)

        elif self.hparams.dataset_type == 'memory_depth':
            if self.hparams.add_random_rays:
                from gp_nerf.datasets.memory_dataset_depth_random import MemoryDataset
            else:
                from gp_nerf.datasets.memory_dataset_depth import MemoryDataset
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
        elif self.hparams.dataset_type == 'llff':
            from gp_nerf.datasets.llff import NeRFDataset
            dataset = NeRFDataset(self.hparams, device=self.device, type='train')
            self.H = dataset.H
            self.W = dataset.W
        elif self.hparams.dataset_type == 'llff_sa3d':
            self.predictor = init_predictor(self.device)
            if self.hparams.sa3d_whole_image:
                from gp_nerf.datasets.llff_sa3d_whole_image import NeRFDataset
            else:
                from gp_nerf.datasets.llff_sa3d_N import NeRFDataset
            dataset = NeRFDataset(self.hparams, device=self.device, type='train', predictor=self.predictor)
            self.H = dataset.H
            self.W = dataset.W
        elif self.hparams.dataset_type == 'mega_sa3d':
            self.predictor = init_predictor(self.device)
            from gp_nerf.datasets.mega_sa3d_whole_image import MemoryDataset_SAM_sa3d
            dataset = MemoryDataset_SAM_sa3d(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams, predictor=self.predictor)
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
                # for i in  range(30):
                #     dataset.load_chunk()
                #     chunk_id += 1
                #     print(f"chunk_id: {chunk_id}")

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
                
                if item == None:
                    continue
                elif item == ['end']:
                    self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index,
                                  dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)
                    val_metrics = self._run_validation(train_iterations)
                    self._write_final_metrics(val_metrics, train_iterations)
                    # raise TypeError

                # item = np.load('79972.npy',allow_pickle=True).item()

                if dataset_index <= discard_index:
                    continue
                discard_index = -1

                # amp: Automatic mixed precision
                with torch.cuda.amp.autocast(enabled=self.hparams.amp):

                    # normal loss
                    if self.hparams.normal_loss:
                        item['c2w_rots'] = c2w_rots
                    # depth loss
                    if self.hparams.dataset_type == 'memory_depth':
                        # print(item['rays'].size())
                        self.train_img_num = item['rgbs'].shape[0]
                    # add random_rays when depth
                    if self.hparams.add_random_rays:
                        self.sample_random_num_current = item['rays'].shape[0] * self.hparams.sample_random_num_each
                    

                    # 调整shape
                    if self.hparams.enable_semantic:
                        for key in item.keys():
                            if item[key].dim() != 1:
                                if item[key].shape[-1] == 1:
                                    item[key] = item[key].reshape(-1)
                                else:
                                    item[key] = item[key].reshape(-1, item[key].shape[-1])

                            # if item[key].dim() == 2:
                            #     item[key] = item[key].reshape(-1)
                            # elif item[key].dim() == 3:
                            #     item[key] = item[key].reshape(-1, *item[key].shape[2:])
                        for key in item.keys():
                            if 'random' in key:
                                continue
                            elif 'random_'+key in item.keys():
                                item[key] = torch.cat((item[key], item['random_'+key]))

                    if self.hparams.enable_semantic and 'labels' in item.keys():
                        labels = item['labels'].to(self.device, non_blocking=True)
                        # if self.hparams.dataset_type != 'sam_project':
                            # from tools.unetformer.uavid2rgb import remapping
                            # labels = remapping(labels)
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
                    val_metrics = self._run_validation(train_iterations)
                    if 'llff' not in self.hparams.dataset_type and 'sa3d' not in self.hparams.dataset_type:
                        self._write_final_metrics(val_metrics, train_iterations)
                
                
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
        val_metrics = self._run_validation(train_iterations)
        self._write_final_metrics(val_metrics, train_iterations=train_iterations)
        


    def _write_final_metrics(self, val_metrics: Dict[str, float], train_iterations) -> None:
        if self.is_master:
            experiment_path_current = self.experiment_path / "eval_{}".format(train_iterations)
            with (experiment_path_current /'metrics.txt').open('a') as f:
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
            depth_scale = item['depth_scale'].to(self.device, non_blocking=True)
            gt_depths = item['depth_dji'].to(self.device, non_blocking=True)
        else:
            depth_scale = None
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
                                                        depth_scale=depth_scale,
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
            pred_depths = results['depth_fine'] * depth_scale
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
                # sigma_loss = -torch.log(weights + 1e-5) * torch.exp(-(z_vals - (gt_depths_valid / (depth_scale[valid_depth_mask]))[:,None]) ** 2 / (2 * err)) * dists
                # sigma_loss = torch.sum(sigma_loss, dim=1).mean()
                # metrics['sigma_loss'] = sigma_loss

                metrics['sigma_loss'] = results['sigma_loss']
                metrics['loss'] += self.hparams.wgt_sigma_loss * sigma_loss




        #semantic loss
        if self.hparams.enable_semantic:
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
                            # self.writer.add_histogram("gt_labels", labels,train_iterations)
                            # self.writer.add_histogram("pred_labels", sem_label,train_iterations)
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
            
                
        if self.hparams.depth_loss or self.hparams.normal_loss:
            decay_min = 0.00 # arrive decay_min in 1/4 training iterationsc
            nloss_decay = (1 - decay_min) * (1 - train_iterations / (self.hparams.train_iterations/4.)) + decay_min
            nloss_decay = max(nloss_decay, 0)
            fg_mask = (results['bg_lambda_fine'].detach() < 0.01)
            if self.hparams.add_random_rays:
                fg_mask = fg_mask[:-self.sample_random_num_current]

            if self.hparams.depth_loss:
                batch_size = self.train_img_num # batch_size is the number of image
                # TODO: add depth scale
                gt_depths = item['depths'].to(self.device, non_blocking=True)
                if self.hparams.add_random_rays:
                    pred_depths = results['depth_fine'][:-self.sample_random_num_current] * item['depth_scale']
                else:
                    pred_depths = results['depth_fine'] * item['depth_scale']
                ray_num = pred_depths.shape[0] // batch_size# + self.hparams.sample_random_num
                sqrt_num = int(np.sqrt(ray_num))
                if sqrt_num**2 != int(ray_num):
                    raise Exception('ray_num not N*N, %d %d' % (ray_num, sqrt_num))
                depth_loss = self.depth_loss(pred_depths.reshape(batch_size, sqrt_num, sqrt_num), 
                                            (gt_depths * 50 + 0.5).reshape(batch_size, sqrt_num, sqrt_num), 
                                            fg_mask.reshape(batch_size, sqrt_num, sqrt_num))
                metrics['depth_loss'] = depth_loss
                metrics['loss'] += nloss_decay * self.hparams.wgt_depth_loss * depth_loss

            if self.hparams.normal_loss:
                gt_normals = item['normals'].to(self.device, non_blocking=True)
                w2c_rots = item['w2c_rots']
                w2c_rot = w2c_rots[image_indices.long()]
                pred_normals = results[f'normal_map_fine']
                cam_normals = torch.bmm(w2c_rot, pred_normals[:,:,None])[:, :, 0]
                if 'sdf' not in self.hparams.network_type:
                    cam_normals = cam_normals * -1 # flip x, y, z axis
                if fg_mask.sum() > 0:
                    normal_l1 = get_normal_loss(cam_normals[fg_mask], gt_normals[fg_mask])
                    metrics['n_l1_loss'] = normal_l1
                    metrics['loss'] += nloss_decay * self.hparams.wgt_nl1_loss * normal_l1
                else:
                    metrics['n_l1_loss'] = metrics['n_cos_loss'] = 0

            if self.writer is not None:
                self.writer.add_scalar('1_train/nloss_decay', nloss_decay, train_iterations)


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

    def render_zyq(self):
        if self.hparams.ckpt_path is not None:
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            train_iterations = checkpoint['iteration']
        self._setup_experiment_dir()
        val_metrics = self._run_validation_render_zyq(train_iterations)
        self._write_final_metrics(val_metrics, train_iterations=train_iterations)
    
    def _run_validation_render_zyq(self, train_index=-1) -> Dict[str, float]:

        with torch.inference_mode():
            #semantic 
            self.metrics_val = Evaluator(num_class=self.hparams.num_semantic_classes)
            CLASSES = ('Cluster', 'Building', 'Road', 'Car', 'Tree', 'Vegetation', 'Human', 'Sky', 'Water', 'Ground', 'Mountain')
            self.nerf.eval()
            val_metrics = defaultdict(float)
            base_tmp_path = None
            
            dataset_path = Path(self.hparams.dataset_path)

            val_paths = sorted(list((dataset_path / 'render_far' / 'metadata').iterdir()))
            # val_paths = sorted(list((dataset_path / 'render_line' / 'metadata').iterdir()))

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

            indices_to_eval = np.arange(len(render_items))
            
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
                    metadata_item = render_items[i]
                    i = int(Path(metadata_item.depth_dji_path).stem)
                    self.hparams.sampling_mesh_guidance = False
                    results, _ = self.render_image(metadata_item, train_index)
                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    
                    # get rendering rgbs and depth
                    viz_result_rgbs = results[f'rgb_{typ}'].view(H, W, 3).cpu()
                    viz_result_rgbs = viz_result_rgbs.clamp(0,1)
                    

                    if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'pred_rgb')):
                        Path(str(experiment_path_current / 'val_rgbs' / 'pred_rgb')).mkdir()
                    Image.fromarray((viz_result_rgbs.numpy() * 255).astype(np.uint8)).save(
                        str(experiment_path_current / 'val_rgbs' / 'pred_rgb' / ("%06d.jpg" % i)))
                    
                    
                        
                    # NOTE: 这里初始化了一个list，需要可视化的东西可以后续加上去
                    img_list = [viz_result_rgbs * 255]

                    
                    prepare_depth_normal_visual(img_list, self.hparams, metadata_item, typ, results, Runner.visualize_scalars, experiment_path_current, i)

                    get_semantic_gt_pred_render_zyq(results, 'val', metadata_item, viz_result_rgbs, self.logits_2_label, typ, remapping,
                                        self.metrics_val, self.metrics_val_each, img_list, experiment_path_current, i, self.writer, self.hparams)
                    
                    # NOTE: 对需要可视化的list进行处理
                    # save images: list：  N * (H, W, 3),  -> tensor(N, 3, H, W)
                    # 将None元素转换为zeros矩阵
                    img_list = [torch.zeros_like(viz_result_rgbs) if element is None else element for element in img_list]
                    img_list = torch.stack(img_list).permute(0,3,1,2)
                    img = make_grid(img_list, nrow=3)
                    img_grid = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    Image.fromarray(img_grid).save(str(experiment_path_current / 'val_rgbs' / ("%06d_all.jpg" % i)))

                    del results
            
            return val_metrics
            
    
    def _run_validation(self, train_index=-1) -> Dict[str, float]:
        if 'llff' in self.hparams.dataset_type:
            from gp_nerf.rendering_gpnerf import render_rays
            with torch.inference_mode():
                
                from gp_nerf.datasets.llff import NeRFDataset
                dataset = NeRFDataset(self.hparams, device=self.device, type='test')
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0,
                                                pin_memory=False)
                
                
                experiment_path_current = self.experiment_path / "eval_{}".format(train_index)
                Path(str(experiment_path_current)).mkdir()
                Path(str(experiment_path_current / 'val_rgbs')).mkdir()
                for dataset_index, item in enumerate(data_loader): #, start=10462):
                    #semantic 
                    if self.hparams.enable_semantic:
                        for key in item.keys():
                            if item[key].dim() == 2:
                                item[key] = item[key].reshape(-1)
                            elif item[key].dim() == 3:
                                item[key] = item[key].reshape(-1, *item[key].shape[2:])
                        for key in item.keys():
                            if 'random' in key:
                                continue
                            elif 'random_'+key in item.keys():
                                item[key] = torch.cat((item[key], item['random_'+key]))
                    
                    

                    
                    rgbs = item['rgbs'].to(self.device, non_blocking=True)
                    groups = None
                    
                    results = {}

                    rays = item['rays'].to(self.device, non_blocking=True)
                    image_indices = item['img_indices'].to(self.device, non_blocking=True)
                    # print(labels.shape[0])
                    for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
                        result_batch, _ = render_rays(nerf=self.nerf, bg_nerf=self.bg_nerf,
                                        rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                        image_indices=image_indices[i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                                        hparams=self.hparams,
                                        sphere_center=self.sphere_center,
                                        sphere_radius=self.sphere_radius,
                                        get_depth=True,
                                        get_depth_variance=False,
                                        get_bg_fg_rgb=True,
                                        train_iterations=train_index)
                                        

                        for key, value in result_batch.items():
                            if key not in results:
                                results[key] = []
                            results[key].append(value.cpu())

                    for key, value in results.items():
                        results[key] = torch.cat(value)
                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    viz_result_rgbs = results[f'rgb_{typ}'].view(dataset.H, dataset.W, 3).cpu()
                    # Image.fromarray(viz_result_rgbs*255).save(str(experiment_path_current / 'val_rgbs' / ("%06d_all.jpg" % dataset_index)))
                    # viz_result_rgbs = (viz_result_rgbs.numpy()*255)[:,:,::-1]
                    # viz_result_rgbs = viz_result_rgbs[:,:,::-1]
                    if 'rgbs' in item:
                        viz_rgbs = item['rgbs'].view(*viz_result_rgbs.shape)
                        img_list = [viz_rgbs * 255, viz_result_rgbs * 255]

                        image_diff = np.abs(viz_rgbs.numpy() - viz_result_rgbs.numpy()).mean(2) # .clip(0.2) / 0.2
                        image_diff_color = cv2.applyColorMap((image_diff*255).astype(np.uint8), cv2.COLORMAP_JET)
                        image_diff_color = cv2.cvtColor(image_diff_color, cv2.COLOR_RGB2BGR)
                        img_list.append(torch.from_numpy(image_diff_color))
                    else:
                        img_list = [viz_result_rgbs * 255]

                    
                    
                    # cv2.imwrite(str(experiment_path_current / 'val_rgbs' / ("%06d_all.jpg" % dataset_index)), viz_result_rgbs)
                    # cv2.imwrite(str(experiment_path_current / 'val_rgbs' / ("%06d_gt.jpg" % dataset_index)), (item['rgbs'].view(*viz_result_rgbs.shape).numpy()*255)[:,:,::-1])
                    
                    if self.hparams.enable_semantic:
                        if f'sem_map_{typ}' in results:
                            sem_logits = results[f'sem_map_{typ}']
                            if self.hparams.use_mask_type == 'hashgrid_mlp':
                                sem_logits = self.nerf.mask_fc_hash(sem_logits.to(self.device)).cpu()
                            elif self.hparams.use_mask_type == 'densegrid_mlp':
                                sem_logits = self.nerf.mask_fc_dense(sem_logits.to(self.device)).cpu()

                            if self.hparams.dataset_type == 'llff':
                                sem_label = self.logits_2_label(sem_logits)
                                visualize_sem = custom2rgb(sem_label.view(*viz_result_rgbs.shape[:-1]).cpu().numpy())
                                img_list.append(torch.from_numpy(visualize_sem))
                            else:
                                colorize_mask = torch.zeros((self.H, self.W, 3))
                                for seg_idx in range(sem_logits.shape[-1]):
                                    if self.hparams.num_semantic_classes <5:
                                        img_list.append((sem_logits[:,seg_idx]>0).view(*viz_result_rgbs.shape[:-1],1).repeat(1,1,3).cpu()*255)
                                    colorize_mask[(sem_logits[:,seg_idx]>0).view(*viz_result_rgbs.shape[:-1])] = self.color_list[seg_idx] #torch.randint(0, 255,(3,)).to(torch.float32)
                                img_list.append(colorize_mask)
                    if f'depth_{typ}' in results:
                        depth_map = results[f'depth_{typ}']
                        # if f'fg_depth_{typ}' in results:
                        #     to_use = results[f'fg_depth_{typ}'].view(-1)
                        #     while to_use.shape[0] > 2 ** 24:
                        #         to_use = to_use[::2]
                        #     ma = torch.quantile(to_use, 0.95)
                        #     depth_clamp = depth_map.clamp_max(ma)
                        # else:
                        #     depth_clamp = depth_map
                        depth_clamp = depth_map

                        depth_vis = torch.from_numpy(Runner.visualize_scalars(
                            torch.log(depth_clamp + 1e-8).view(*viz_result_rgbs.shape[:-1]).cpu()))
                        img_list.append(depth_vis)
                    
                    
                    img_list = [torch.zeros_like(viz_result_rgbs) if element is None else element for element in img_list]
                    img_list = torch.stack(img_list).permute(0,3,1,2)
                    img = make_grid(img_list, nrow=3)
                    img_grid = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    Image.fromarray(img_grid).save(str(experiment_path_current / 'val_rgbs' / ("%06d_all.jpg" % dataset_index)))

        elif self.hparams.dataset_type == 'mega_sa3d':
            with torch.inference_mode():
                val_type = self.hparams.val_type  # train  val
                print('val_type: ', val_type)
                if val_type == 'val':
                    if 'residence'in self.hparams.dataset_path:
                        self.val_items=self.val_items[:19]
                    elif 'building'in self.hparams.dataset_path or 'campus'in self.hparams.dataset_path:
                        self.val_items=self.val_items[:10]
                    # self.val_items=self.val_items[:2]
                    indices_to_eval = np.arange(len(self.val_items))
                elif val_type == 'train':
                    # #indices_to_eval = np.arange(0, len(self.train_items), 100)  
                    # indices_to_eval = [0] #np.arange(len(self.train_items)) 
                    indices_to_eval = np.arange(415,490)  
                
                experiment_path_current = self.experiment_path / "eval_{}".format(train_index)
                Path(str(experiment_path_current)).mkdir()
                Path(str(experiment_path_current / 'val_rgbs')).mkdir()

                for i in main_tqdm(indices_to_eval):
                    if val_type == 'val':
                        metadata_item = self.val_items[i]
                    elif val_type == 'train':
                        metadata_item = self.train_items[i]

                    results, _ = self.render_image(metadata_item, train_index)

                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    viz_rgbs = metadata_item.load_image().float() / 255.
                    self.H, self.W = viz_rgbs.shape[0], viz_rgbs.shape[1]
                    viz_result_rgbs = results[f'rgb_{typ}'].view(self.H, self.W, 3).cpu()
                    # viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()

                    # Image.fromarray(viz_result_rgbs*255).save(str(experiment_path_current / 'val_rgbs' / ("%06d_all.jpg" % i)))
                    # viz_result_rgbs = (viz_result_rgbs.numpy()*255)[:,:,::-1]
                    # viz_result_rgbs = viz_result_rgbs[:,:,::-1]
                    img_list = [viz_result_rgbs * 255]

                    
                    
                    # cv2.imwrite(str(experiment_path_current / 'val_rgbs' / ("%06d_all.jpg" % i)), viz_result_rgbs)
                    # cv2.imwrite(str(experiment_path_current / 'val_rgbs' / ("%06d_gt.jpg" % i)), (item['rgbs'].view(*viz_result_rgbs.shape).numpy()*255)[:,:,::-1])
                    
                    if self.hparams.enable_semantic:
                        if f'sem_map_{typ}' in results:
                            sem_logits = results[f'sem_map_{typ}']

                            if self.hparams.use_mask_type == 'hashgrid_mlp':
                                sem_logits = self.nerf.mask_fc_hash(sem_logits.to(self.device)).cpu()
                            elif self.hparams.use_mask_type == 'densegrid_mlp':
                                sem_logits = self.nerf.mask_fc_dense(sem_logits.to(self.device)).cpu()
                            if self.hparams.dataset_type == 'llff':
                                sem_label = self.logits_2_label(sem_logits)
                                visualize_sem = custom2rgb(sem_label.view(*viz_result_rgbs.shape[:-1]).cpu().numpy())
                                img_list.append(torch.from_numpy(visualize_sem))
                            else:
                                colorize_mask = torch.zeros((self.H, self.W, 3))
                                for seg_idx in range(sem_logits.shape[-1]):
                                    if self.hparams.num_semantic_classes <10:
                                        img_list.append((sem_logits[:,seg_idx]>0).view(*viz_result_rgbs.shape[:-1],1).repeat(1,1,3).cpu()*255)
                                    colorize_mask[(sem_logits[:,seg_idx]>0).view(*viz_result_rgbs.shape[:-1])] = self.color_list[seg_idx]  # torch.randint(0, 255,(3,)).to(torch.float32)
                                    # if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'each')):
                                    #     Path(str(experiment_path_current / 'val_rgbs' / 'each')).mkdir()
                                    # Image.fromarray((colorize_mask.numpy() * 255).astype(np.uint8)).save(
                                    #     str(experiment_path_current / 'val_rgbs' / 'each' / ("%06d_colorize_mask_%01d.jpg" % (i, seg_idx))))
                                    
                                img_list.append(colorize_mask)
                                if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'colorize_mask')):
                                    Path(str(experiment_path_current / 'val_rgbs' / 'colorize_mask')).mkdir()
                                Image.fromarray((colorize_mask.numpy() * 255).astype(np.uint8)).save(
                                    str(experiment_path_current / 'val_rgbs' / 'colorize_mask' / ("%06d_colorize_mask.jpg" % i)))
                                

                    if f'depth_{typ}' in results:
                        depth_map = results[f'depth_{typ}']
                        # if f'fg_depth_{typ}' in results:
                        #     to_use = results[f'fg_depth_{typ}'].view(-1)
                        #     while to_use.shape[0] > 2 ** 24:
                        #         to_use = to_use[::2]
                        #     ma = torch.quantile(to_use, 0.95)
                        #     depth_clamp = depth_map.clamp_max(ma)
                        # else:
                        #     depth_clamp = depth_map
                        depth_clamp = depth_map

                        depth_vis = torch.from_numpy(Runner.visualize_scalars(
                            torch.log(depth_clamp + 1e-8).view(*viz_result_rgbs.shape[:-1]).cpu()))
                        img_list.append(depth_vis)
                    
                    
                    img_list = [torch.zeros_like(viz_result_rgbs) if element is None else element for element in img_list]
                    img_list = torch.stack(img_list).permute(0,3,1,2)
                    img = make_grid(img_list, nrow=3)
                    img_grid = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    Image.fromarray(img_grid).save(str(experiment_path_current / 'val_rgbs' / ("%06d_all.jpg" % i)))

        else:
            if 'residence'in self.hparams.dataset_path or 'campus'in self.hparams.dataset_path:
                from tools.unetformer.uavid2rgb import remapping_remove_ground as remapping
            else:
                from tools.unetformer.uavid2rgb import remapping

            if self.hparams.use_neus_gradient == False and self.hparams.nr3d_nablas == False:
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
                            elif 'building'in self.hparams.dataset_path or 'campus'in self.hparams.dataset_path:
                                self.val_items=self.val_items[:10]
                            # self.val_items=self.val_items[:2]
                            indices_to_eval = np.arange(len(self.val_items))
                        elif val_type == 'train':
                            # #indices_to_eval = np.arange(0, len(self.train_items), 100)  
                            # indices_to_eval = [0] #np.arange(len(self.train_items))  
                            # indices_to_eval = np.arange(450,510)  
                            # indices_to_eval = np.arange(370,800)  
                            indices_to_eval = np.arange(370,490)  
                            
                        
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
                            # samantic_each_value['OA'] = []

                            
                            for i in main_tqdm(indices_to_eval):
                                self.metrics_val_each = Evaluator(num_class=self.hparams.num_semantic_classes)
                                # if i != 0:
                                #     break
                                if val_type == 'val':
                                    metadata_item = self.val_items[i]
                                elif val_type == 'train':
                                    metadata_item = self.train_items[i]
                                
                                results, _ = self.render_image(metadata_item, train_index)
                                typ = 'fine' if 'rgb_fine' in results else 'coarse'
                                
                                viz_rgbs = metadata_item.load_image().float() / 255.
                                if self.hparams.save_depth:
                                    save_depth_dir = os.path.join(str(self.experiment_path), "depth_{}".format(train_index))
                                    if not os.path.exists(save_depth_dir):
                                        os.makedirs(save_depth_dir)
                                    depth_map = results[f'depth_{typ}'].view(viz_rgbs.shape[0], viz_rgbs.shape[1]).numpy().astype(np.float16)
                                    np.save(os.path.join(save_depth_dir, metadata_item.image_path.stem + '.npy'), depth_map)
                                    continue
                                
                                # get rendering rgbs and depth
                                viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()
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
                                
                                get_semantic_gt_pred(results, val_type, metadata_item, viz_rgbs, self.logits_2_label, typ, remapping,
                                                    self.metrics_val, self.metrics_val_each, img_list, experiment_path_current, i, self.writer, self.hparams, viz_result_rgbs * 255)
                                    

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
                                        
                                        if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'fg_bg')) and self.hparams.save_individual:
                                            Path(str(experiment_path_current / 'val_rgbs' / 'fg_bg')).mkdir()
                                        if self.hparams.save_individual:
                                            img.save(str(experiment_path_current / 'val_rgbs' / 'fg_bg' / ("%06d_fg_bg.jpg" % i)))
                                    
                                    # logger
                                    samantic_each_value = save_semantic_metric(self.metrics_val_each, CLASSES, samantic_each_value, self.wandb, self.writer, train_index, i)
                                    self.metrics_val_each.reset()
                                del results
                        # logger
                        write_metric_to_folder_logger(self.metrics_val, CLASSES, experiment_path_current, samantic_each_value, self.wandb, self.writer, train_index, self.hparams)
                        self.metrics_val.reset()
                        
                        self.writer.flush()
                        self.writer.close()
                        self.nerf.train()
                    finally:
                        if self.is_master and base_tmp_path is not None:
                            shutil.rmtree(base_tmp_path)

                    return val_metrics
            else:
                # with torch.inference_mode():
                with torch.no_grad():
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
                            elif 'building'in self.hparams.dataset_path or 'campus'in self.hparams.dataset_path:
                                self.val_items=self.val_items[:10]
                            # self.val_items=self.val_items[:2]
                            indices_to_eval = np.arange(len(self.val_items))
                        elif val_type == 'train':
                            indices_to_eval = np.arange(800,1200)  


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
                            # samantic_each_value['OA'] = []

                            
                            for i in main_tqdm(indices_to_eval):
                                self.metrics_val_each = Evaluator(num_class=self.hparams.num_semantic_classes)
                                # if i != 0:
                                #     break
                                if val_type == 'val':
                                    metadata_item = self.val_items[i]
                                elif val_type == 'train':
                                    metadata_item = self.train_items[i]
                                viz_rgbs = metadata_item.load_image().float() / 255.
                                
                                results, _ = self.render_image(metadata_item, train_index)
                                typ = 'fine' if 'rgb_fine' in results else 'coarse'

                                if self.hparams.save_depth:
                                    save_depth_dir = os.path.join(str(self.experiment_path), "depth_{}".format(train_index))
                                    if not os.path.exists(save_depth_dir):
                                        os.makedirs(save_depth_dir)
                                    depth_map = results[f'depth_{typ}'].view(viz_rgbs.shape[0], viz_rgbs.shape[1]).numpy().astype(np.float16)
                                    np.save(os.path.join(save_depth_dir, metadata_item.image_path.stem + '.npy'), depth_map)
                                    continue
                                
                                # get rendering rgbs and depth
                                viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()
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
                                

                                get_semantic_gt_pred(results, val_type, metadata_item, viz_rgbs, self.logits_2_label, typ, remapping,
                                                    self.metrics_val, self.metrics_val_each, img_list, experiment_path_current, i, self.writer, self.hparams)
                                   
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
                                

                                if val_type == 'val':
                                    #save  [pred_label, pred_rgb, fg_bg] to the folder 

                                    if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'gt_rgb')) and self.hparams.save_individual:
                                        Path(str(experiment_path_current / 'val_rgbs' / 'gt_rgb')).mkdir()
                                    if self.hparams.save_individual:
                                        Image.fromarray((viz_rgbs.numpy() * 255).astype(np.uint8)).save(
                                            str(experiment_path_current / 'val_rgbs' / 'gt_rgb' / ("%06d_gt_rgb.jpg" % i)))

                                    if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'pred_rgb')):
                                        Path(str(experiment_path_current / 'val_rgbs' / 'pred_rgb')).mkdir()
                                    Image.fromarray((viz_result_rgbs.numpy() * 255).astype(np.uint8)).save(
                                        str(experiment_path_current / 'val_rgbs' / 'pred_rgb' / ("%06d_pred_rgb.jpg" % i)))
                                    

                                    if self.hparams.bg_nerf or f'bg_rgb_{typ}' in results:
                                        img = Runner._create_fg_bg_image(results[f'fg_rgb_{typ}'].view(viz_rgbs.shape[0],viz_rgbs.shape[1], 3).cpu(),
                                                                        results[f'bg_rgb_{typ}'].view(viz_rgbs.shape[0],viz_rgbs.shape[1], 3).cpu())
                                        
                                        if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'fg_bg')) and self.hparams.save_individual:
                                            Path(str(experiment_path_current / 'val_rgbs' / 'fg_bg')).mkdir()
                                        if self.hparams.save_individual:
                                        
                                            img.save(str(experiment_path_current / 'val_rgbs' / 'fg_bg' / ("%06d_fg_bg.jpg" % i)))
                                    
                                    # logger
                                    samantic_each_value = save_semantic_metric(self.metrics_val_each, CLASSES, samantic_each_value, self.wandb, self.writer, train_index, i)
                                    self.metrics_val_each.reset()
                                del results
                        # logger
                        write_metric_to_folder_logger(self.metrics_val, CLASSES, experiment_path_current, samantic_each_value, self.wandb, self.writer, train_index)
                        self.metrics_val.reset()
                        
                        self.writer.flush()
                        self.writer.close()
                        self.nerf.train()
                    finally:
                        if self.is_master and base_tmp_path is not None:
                            shutil.rmtree(base_tmp_path)

                    return val_metrics

    def _run_validation_project_val_points(self, train_index=-1) -> Dict[str, float]:
        self._setup_experiment_dir()


        with torch.inference_mode():
            #semantic 
            self.nerf.eval()
            val_type = self.hparams.val_type  # train  val
            print('val_type: ', val_type)

            indices_to_eval = np.arange(len(self.val_items))
            

            experiment_path_current = self.experiment_path / "eval_{}".format(train_index)
            Path(str(experiment_path_current)).mkdir()
            Path(str(experiment_path_current / 'val_rgbs')).mkdir()
            
            for i in indices_to_eval[5:6]:
                if val_type == 'val':
                    metadata_item = self.val_items[i]
                elif val_type == 'train':
                    metadata_item = self.train_items[i]
                viz_rgbs = metadata_item.load_image().float() / 255.

                save_dir = f'zyq/val_{metadata_item.image_path.stem}_v4'
                print('save_dir %s' % save_dir)
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                results, _ = self.render_image(metadata_item, train_index)
                typ = 'fine' if 'rgb_fine' in results else 'coarse'
                viz_rgbs = metadata_item.load_image().float() / 255.
                
                depth_map = results[f'depth_{typ}'].view(viz_rgbs.shape[0], viz_rgbs.shape[1])

                H, W = depth_map.shape

                directions = get_ray_directions(W,
                                                H,
                                                metadata_item.intrinsics[0],
                                                metadata_item.intrinsics[1],
                                                metadata_item.intrinsics[2],
                                                metadata_item.intrinsics[3],
                                                self.hparams.center_pixels,
                                                'cpu')
                depth_scale = torch.abs(directions[:, :, 2]) # z-axis's values
                depth_map = (depth_map * depth_scale).numpy()

                x, y = int(W * 1/2), int(H * 1/2)
                #x, y = W // 2, H // 2
                K1 = metadata_item.intrinsics
                #K1 = np.array([[K1[0], 0, K1[2]],[0, -K1[1], K1[3]],[0,0,-1]])
                K1 = np.array([[K1[0], 0, K1[2]],[0, K1[1], K1[3]],[0,0,1]])
                E1 = np.array(metadata_item.c2w)
                E1 = np.stack([E1[:, 0], E1[:, 1]*-1, E1[:, 2]*-1, E1[:, 3]], 1)
                print('camera c2w', metadata_item.c2w)
                print('camera intrinsic', metadata_item.intrinsics)
                depth = depth_map[y, x]   #* 0.5
                pt_3d = np.dot(np.linalg.inv(K1), np.array([x, y, 1])) * depth
                pt_3d = np.append(pt_3d, 1)
                print('3d point in camera space', pt_3d)
                (Path(f"{save_dir}")).mkdir(exist_ok=True)
                (Path(f"{save_dir}/occluded")).mkdir(exist_ok=True)
                img1= cv2.imread(f"{str(metadata_item.image_path)}")
                pt2 = (int(x), int(y))
                radius = 5
                color = (0, 0, 255)
                thickness = 2
                cv2.circle(img1, pt2, radius, color, thickness)
                cv2.imwrite(f"{save_dir}/000000_{metadata_item.image_path.stem}.jpg", img1)
                
                world_point = np.dot(np.concatenate((E1, [[0,0,0,1]]), 0), pt_3d)
                print('point in world ', world_point)

                #for j in [30]:
                for j in np.arange(len(self.train_items[:])):

                    """ Compute occlusion """

                    train_item = self.train_items[j]


                    E2 = np.array(train_item.c2w)
                    E2 = np.stack([E2[:, 0], E2[:, 1]*-1, E2[:, 2]*-1, E2[:, 3]], 1)
                    w2c = np.linalg.inv(np.concatenate((E2, [[0,0,0,1]]), 0))
                    pt_3d_trans = np.dot(w2c, world_point)

                    pt_2d_trans = np.dot(K1, pt_3d_trans[:3]) 
                    pt_2d_trans = pt_2d_trans / pt_2d_trans[2]
                    print('%d, camera point in %s' % (j, train_item.image_path.stem), pt_3d_trans, pt_2d_trans)
                    
                    h2, w2 = H, W
                    x2, y2 = int(pt_2d_trans[0]), int(pt_2d_trans[1])
                    if x2 >= 0 and x2 < w2 and y2 >= 0 and y2 < h2:
                        results2, _ = self.render_image(train_item, train_index)
                        depth_map2 = results2[f'depth_{typ}'].view(viz_rgbs.shape[0], viz_rgbs.shape[1])
                        depth_map2 = (depth_map2 * depth_scale).numpy()
                        depth2 = depth_map2[y2, x2]
                        img2 = cv2.imread(str(train_item.image_path))
                        pt2 = (int(pt_2d_trans[0]), int(pt_2d_trans[1]))
                        depth_diff = np.abs(depth2 - pt_3d_trans[2])
                        print('Project inside image 2', depth, depth2, depth_diff)
                        radius = 5
                        color = (0, 0, 255)
                        thickness = 2
                        if depth_diff > 0.01:
                            print('occluded points!!')
                            color = (0, 255, 0)
                            img2 = cv2.circle(img2, pt2, radius, color, thickness)
                            cv2.imwrite(f"{save_dir}/occluded/{train_item.image_path.stem}.jpg", img2)
                        else:
                            img2 = cv2.circle(img2, pt2, radius, color, thickness)
                            cv2.imwrite(f"{save_dir}/{train_item.image_path.stem}.jpg", img2)
                    # else:
                        # print('Projected point is not on image2')
                        

                        
        return -1

    def _run_validation_save_depth(self, train_index=-1) -> Dict[str, float]:
        self._setup_experiment_dir()
        if self.hparams.dataset_type == 'llff':
            self.nerf.eval()
            from gp_nerf.rendering_gpnerf import render_rays
            with torch.inference_mode():
                from gp_nerf.datasets.llff import NeRFDataset
                dataset = NeRFDataset(self.hparams, device=self.device, type='all')
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=0,
                                                pin_memory=False)
                experiment_path_current = self.experiment_path / "eval_{}".format(train_index)
                Path(str(experiment_path_current)).mkdir()
                Path(str(experiment_path_current / 'val_rgbs')).mkdir()

                save_depth_path = Path(dataset.f_paths[0]).parent.parent / 'depths'
                (save_depth_path).mkdir(exist_ok=True)
                
                for dataset_index, item in enumerate(data_loader): #, start=10462):
                    #semantic 
                    for key in item.keys():
                        if item[key].dim() == 2:
                            item[key] = item[key].reshape(-1)
                        elif item[key].dim() == 3:
                            item[key] = item[key].reshape(-1, *item[key].shape[2:])
                    for key in item.keys():
                        if 'random' in key:
                            continue
                        elif 'random_'+key in item.keys():
                            item[key] = torch.cat((item[key], item['random_'+key]))
                    
                    if self.hparams.enable_semantic:
                        labels = item['labels'].to(self.device, non_blocking=True)
                        if self.hparams.dataset_type == 'sam_project':
                            pass
                        else:
                            from tools.unetformer.uavid2rgb import remapping
                            labels = remapping(labels)
                    else:
                        labels = None

                    
                    rgbs = item['rgbs'].to(self.device, non_blocking=True)
                    groups = None
                    
                    results = {}

                    rays = item['rays'].to(self.device, non_blocking=True)
                    image_indices = item['img_indices'].to(self.device, non_blocking=True)
                    # print(labels.shape[0])
                    for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
                        result_batch, _ = render_rays(nerf=self.nerf, bg_nerf=self.bg_nerf,
                                        rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                        image_indices=image_indices[i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                                        hparams=self.hparams,
                                        sphere_center=self.sphere_center,
                                        sphere_radius=self.sphere_radius,
                                        get_depth=True,
                                        get_depth_variance=False,
                                        get_bg_fg_rgb=True,
                                        train_iterations=train_index)

                        for key, value in result_batch.items():
                            if key not in results:
                                results[key] = []
                            results[key].append(value.cpu())

                    for key, value in results.items():
                        results[key] = torch.cat(value)
                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    viz_result_rgbs = results[f'rgb_{typ}'].view(dataset.H, dataset.W, 3).cpu()
                    # Image.fromarray(viz_result_rgbs*255).save(str(experiment_path_current / 'val_rgbs' / ("%06d_all.jpg" % dataset_index)))
                    viz_result_rgbs = (viz_result_rgbs.numpy()*255)[:,:,::-1]
                    cv2.imwrite(str(experiment_path_current / 'val_rgbs' / ("%06d_all.jpg" % dataset_index)), viz_result_rgbs)
                    cv2.imwrite(str(experiment_path_current / 'val_rgbs' / ("%06d_gt.jpg" % dataset_index)), (item['rgbs'].view(*viz_result_rgbs.shape).numpy()*255)[:,:,::-1])
                    
                    depth_map = results[f'depth_{typ}'].view(viz_result_rgbs.shape[0], viz_result_rgbs.shape[1])
                    np.save(f"{save_depth_path}/{Path(dataset.f_paths[dataset_index]).stem}.npy", depth_map)
                    print(f"{save_depth_path}/{Path(dataset.f_paths[dataset_index]).stem}.npy")

        else:
            with torch.inference_mode():
                #semantic 
                self.nerf.eval()
                val_type = self.hparams.val_type  # train  val
                print('val_type: ', val_type)

                indices_to_eval = np.arange(len(self.train_items))
                
                train_depth = self.train_items[0].image_path.parent.parent.parent / 'train'/ 'depths'
                val_depth = self.train_items[0].image_path.parent.parent.parent / 'val'/ 'depths'
                print(train_depth)
                print(val_depth)

                (train_depth).mkdir(exist_ok=True)
                (val_depth).mkdir(exist_ok=True)
                
                for i in tqdm(indices_to_eval):
                    if i % 50 != 0 :
                        continue
                    metadata_item = self.train_items[i]
                    results, _ = self.render_image(metadata_item, train_index)
                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    viz_rgbs = metadata_item.load_image().float() / 255.
                    depth_map = results[f'depth_{typ}'].view(viz_rgbs.shape[0], viz_rgbs.shape[1])
                    if self.train_items[i].is_val: 
                        np.save(f"{val_depth}/{metadata_item.image_path.stem}.npy", depth_map)
                        print(f"{val_depth}/{metadata_item.image_path.stem}.npy")
                    else:
                        np.save(f"{train_depth}/{metadata_item.image_path.stem}.npy", depth_map)

                    
                            
            return -1


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
            rays = get_rays(directions, metadata.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)

            rays = rays.view(-1, 8).to(self.device, non_blocking=True).cuda()  # (H*W, 8)
            if self.hparams.render_zyq:
                image_indices = 817 * torch.ones(rays.shape[0], device=rays.device)
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
                    gt_depths = metadata.load_depth_dji().view(-1).to(self.device)
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
                                              depth_scale=depth_scale[i:i + self.hparams.image_pixel_batch_size],
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
        if self.hparams.dataset_type == 'sam':
            sam_feature_path = None
            candidate = metadata_path.parent.parent / 'sam_features' / '{}.npy'.format(metadata_path.stem)
            if candidate.exists():
                sam_feature_path = candidate
            return ImageMetadata(image_path, metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                                intrinsics, image_index, None if (is_val and self.hparams.all_val) else mask_path, is_val, label_path, sam_feature_path)
        elif self.hparams.dataset_type == 'sam_project' or self.hparams.dataset_type == 'mega_sa3d':
            sam_feature_path = None
            candidate = metadata_path.parent.parent / 'sam_features' / '{}.npy'.format(metadata_path.stem)
            if candidate.exists():
                sam_feature_path = candidate
            # depth_path = os.path.join(metadata_path.parent.parent, 'mono_cues', '%s_depth.npy' % metadata_path.stem) 
            depth_path = os.path.join(metadata_path.parent.parent, 'depths', '%s.npy' % metadata_path.stem) 
            return ImageMetadata(image_path, metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                                intrinsics, image_index, None if (is_val and self.hparams.all_val) else mask_path, is_val, label_path, sam_feature_path, depth_path=depth_path)
        
        elif self.hparams.normal_loss:
            normal_path = os.path.join(metadata_path.parent.parent, 'mono_cues', '%s_normal.npy' % metadata_path.stem) 
            return ImageMetadata(image_path, metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                                 intrinsics, image_index, None if (is_val and self.hparams.all_val) else mask_path, is_val, label_path, normal_path=normal_path)

        elif self.hparams.depth_loss:
            depth_path = os.path.join(metadata_path.parent.parent, 'mono_cues', '%s_depth.npy' % metadata_path.stem) 
            return ImageMetadata(image_path, metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                                 intrinsics, image_index, None if (is_val and self.hparams.all_val) else mask_path, is_val, label_path, depth_path=depth_path)
        elif self.hparams.dataset_type=='memory_depth_dji':
            if self.hparams.depth_dji_type=='las':
                depth_dji_path = os.path.join(metadata_path.parent.parent, 'depth_dji', '%s.npy' % metadata_path.stem) 
            elif self.hparams.depth_dji_type=='mesh':
                depth_dji_path = os.path.join(metadata_path.parent.parent, 'depth_mesh', '%s.npy' % metadata_path.stem) 
                
            return ImageMetadata(image_path, metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                                 intrinsics, image_index, None if (is_val and self.hparams.all_val) else mask_path, is_val, label_path, depth_dji_path=depth_dji_path)

        else:
            return ImageMetadata(image_path, metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                                intrinsics, image_index, None if (is_val and self.hparams.all_val) else mask_path, is_val, label_path)

    def _get_experiment_path(self) -> Path:
        exp_dir = Path(self.hparams.exp_name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        existing_versions = [int(x.name) for x in exp_dir.iterdir()]
        version = 0 if len(existing_versions) == 0 else max(existing_versions) + 1
        experiment_path = exp_dir / str(version)
        return experiment_path


    
    
    def pretrain_sdf_road_surface(
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



        num_point_each_img = 512
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
            depth_scale = torch.abs(directions[:, :, 2:3]) # z-axis's values
            # depth_map = (depth_map * depth_scale).numpy()   #  这里depth_dji_mesh出来的深度可以直接用，表示z
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
        pts_3d[:,0:1] = pts_3d.mean(0)[0].expand_as(pts_3d[:,0:1])
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
                    samples_in_obj = self.nerf.encoding.space.unnormalize_coords(samples_in_net)
                    
                    # For each sample point, find the track point of the minimum distance (measured in 3D space.)
                    # # [num_points, 1, 3] - [1, num_tracks, 3] = [num_points, num_tracks, 3] -> [num_points, num_tracks] -> [num_samples]
                    # ret_min_dis_in_obj = (samples_in_obj.unsqueeze(-2) - tracks_in_obj.unsqueeze(0)).norm(dim=-1).min(dim=-1)
                    
                    # For each sample point, find the track point of the minimum distance (measure at 2D space. i.e. xoy plane for floor_dim=z)
                    # [num_points, 1, 3] - [1, num_tracks, 3] = [num_points, num_tracks, 3] -> [num_points, num_tracks] -> [num_samples]
                    ret_min_dis_in_obj = (samples_in_obj[..., None, other_dims] - tracks_in_obj[None, ..., other_dims]).norm(dim=-1).min(dim=-1)

                    # For each sample point, current floor'z coordinate of floor_dim
                    floor_at_in_obj = tracks_in_obj[ret_min_dis_in_obj.indices][..., floor_dim] - floor_up_sign * ego_height
                    sdf_gt_in_obj = floor_up_sign * (samples_in_obj[..., floor_dim] - floor_at_in_obj)
                    
                    #---- Convert SDF GT value in real-world object's unit to network's unit
                    sdf_gt = sdf_gt_in_obj / self.nerf.sdf_scale
                    # sdf_gt = sdf_gt_in_obj / 25
                    
                    pred_sdf, feature_vector, pred_nablas = self.nerf.forward_sdf_nablas(samples_in_net, nablas_has_grad=True)
                    pred_sdf = pred_sdf
                    nablas_norm = pred_nablas.norm(dim=-1)
                    loss = F.smooth_l1_loss(pred_sdf, sdf_gt, reduction='mean') + w_eikonal * loss_eikonal_fn(nablas_norm)
                    
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
            depth_scale = torch.abs(directions[:, :, 2:3]) # z-axis's values
            # depth_map = (depth_map * depth_scale).numpy()   #  这里depth_dji_mesh出来的深度可以直接用，表示z
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
    

    def pretrain_sdf_road_surface_3d(
        self, 
        num_iters=1000, num_points=5000, 
        lr=1.0e-4, w_eikonal=1.0e-3, 
        safe_mse = True, clip_grad_val: float = 0.1, optim_cfg = {}, 
        floor_dim: Literal['x','y','z'] = 'x', # The vertical dimension of obj coords
        floor_up_sign: Literal[1, -1]=-1, # [-1] if (-)dim points to sky else [1]
        ego_height: float = 0., # Estimated ego's height from road, in obj coords space
        logger: Logger=None, log_prefix: str=None, debug_param_detail=False):

        # 读取mesh表面， 获取顶点法线，  提取法线方向朝向+x的上表面点
        # mesh = trimesh.load_mesh('/data/yuqi/code/GP-NeRF-semantic/data/mesh_subset_nerfcoor.obj')
        # mesh = trimesh.load_mesh('/data/yuqi/code/GP-NeRF-semantic/data/mesh.obj')
        mesh = trimesh.load_mesh(self.hparams.mesh_path)
        

        
        vertex_normals = mesh.vertex_normals
        pts_3d = mesh.vertices[vertex_normals[:, 0] <= 0]
        upper_surface_normal = vertex_normals[vertex_normals[:, 0] <= 0]
        
        min_altitude = pts_3d.min(0)[0]
        min_altitude = (min_altitude - self.hparams.stretch[0][0]) / (self.hparams.stretch[1][0] - self.hparams.stretch[0][0])   


        # sample_points, face_indices = trimesh.sample.sample_surface(mesh, 10000000)
        # sample_normals = mesh.face_normals[face_indices]
        # pts_3d_2 = sample_points[sample_normals[:, 0] < 0]
        # upper_surface_normal_2 = sample_normals[sample_normals[:, 0] < 0]
        
        # upper_surface_normal_2 = upper_surface_normal_2[(pts_3d_2 >= self.hparams.stretch[0].cpu().numpy()*1.1).all(axis=1) & (pts_3d_2 <= self.hparams.stretch[1].cpu().numpy()*1.1).all(axis=1)]
        # pts_3d_2 = pts_3d_2[(pts_3d_2 >= self.hparams.stretch[0].cpu().numpy()*1.1).all(axis=1) & (pts_3d_2 <= self.hparams.stretch[1].cpu().numpy()*1.1).all(axis=1)]  # 只要bound内的点
        # del sample_points, face_indices, sample_normals, mesh
        # pts_3d = np.concatenate([pts_3d, pts_3d_2], 0)
        # upper_surface_normal = np.concatenate([upper_surface_normal, upper_surface_normal_2], 0)



        # vi = np.concatenate([pts_3d, upper_surface_normal],1)
        # vi = torch.from_numpy(np.array(vi)).to(self.device)
        # visualize_points_list = [vi.cpu().numpy()]
        # visualize_points(visualize_points_list)

        # num_point_each_img = 512
        # pts_3d = []
        # for idx in range(len(self.train_items)):
        #     metadata_item = self.train_items[idx]
        #     depth_map = metadata_item.load_depth_dji()

        #     H, W = depth_map.shape[:2]
        #     directions = get_ray_directions(W,
        #                                 H,
        #                                 metadata_item.intrinsics[0],
        #                                 metadata_item.intrinsics[1],
        #                                 metadata_item.intrinsics[2],
        #                                 metadata_item.intrinsics[3],
        #                                 self.hparams.center_pixels,
        #                                 'cpu')
        #     depth_scale = torch.abs(directions[:, :, 2:3]) # z-axis's values
        #     # depth_map = (depth_map * depth_scale).numpy()   #  这里depth_dji_mesh出来的深度可以直接用，表示z
        #     depth_map = depth_map.numpy()
        #     K1 = metadata_item.intrinsics
        #     K1 = np.array([[K1[0], 0, K1[2]],[0, K1[1], K1[3]],[0,0,1]])
        #     E1 = np.array(metadata_item.c2w)
        #     E1 = np.stack([E1[:, 0], E1[:, 1]*-1, E1[:, 2]*-1, E1[:, 3]], 1)

        #     for p in range(num_point_each_img):
        #         x, y = random.randint(0, W-1), random.randint(0, H-1)
        #         depth = depth_map[y, x]   #* 0.5
        #         if torch.isinf(torch.from_numpy(depth)):
        #             continue
        #         pt_3d = np.dot(np.linalg.inv(K1), np.array([x, y, 1])) * depth
        #         pt_3d = np.append(pt_3d, 1)
        #         world_point = np.dot(np.concatenate((E1, [[0,0,0,1]]), 0), pt_3d)
        #         pts_3d.append(world_point[:3])

        pts_3d = torch.from_numpy(np.array(pts_3d)).to(self.device)
        # pts_3d = pts_3d[(pts_3d[:,1:] >= self.hparams.stretch[0][1:]*1.1).all(axis=1) & (pts_3d[:,1:] <= self.hparams.stretch[1][1:]*1.1).all(axis=1)]  # 只要bound内的点


        


        # visualize_points_list = [pts_3d.cpu().numpy()]
        # visualize_points(visualize_points_list)
        tracks_in_obj = (contract_to_unisphere_new(pts_3d, self.hparams)).float()
        # tracks_in_obj = tracks_in_obj[(tracks_in_obj >= -1).all(axis=1) & (tracks_in_obj <= 1).all(axis=1)]  # 只要bound内的点


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
        
        
        with torch.enable_grad():
            with tqdm(range(num_iters), desc=f"=> Pretraining SDF...") as pbar:
                for it in pbar:
                    # torch.cuda.empty_cache()

                    samples_in_net = torch.empty([num_points,3], dtype=torch.float, device=device).uniform_(-1, 1)
                    # samples_in_obj = self.nerf.encoding.space.unnormalize_coords(samples_in_net)
                    

                    # kdtree_B = cKDTree(tracks_in_obj.cpu().numpy())
                    # nearest_distances, nearest_indices = kdtree_B.query(samples_in_net.cpu().numpy(), k=1)
                    
                    nearest_distances = torch.cdist(samples_in_net, tracks_in_obj)
                    nearest_indices = torch.argmin(nearest_distances, dim=1)


                    # 计算与最近点的距离， 符号由法线的朝向决定， 这里负值向天空
                    floor_at_in_obj = tracks_in_obj[nearest_indices]
                    sdf_gt_in_obj = (samples_in_net - floor_at_in_obj).norm(dim=-1)


                    derection_vector = (samples_in_net - floor_at_in_obj)
                    mesh_normal = torch.from_numpy(upper_surface_normal[nearest_indices.cpu().numpy()]).to(self.device)
                    dot_product = torch.sum(derection_vector * mesh_normal, axis=1)
                    sign_adjust = torch.where((dot_product > 0) | (samples_in_net[:,0] < min_altitude), 1, -1)
                    # sign_adjust = torch.where(dot_product > 0, 1, -1)
                    sdf_gt_in_obj = sign_adjust * sdf_gt_in_obj


                    #---- Convert SDF GT value in real-world object's unit to network's unit
                    sdf_gt = sdf_gt_in_obj / self.nerf.sdf_scale
                    # sdf_gt = sdf_gt_in_obj / 25
                    
                    # 1. nablas
                    pred_sdf, feature_vector, pred_nablas = self.nerf.forward_sdf_nablas(samples_in_net, nablas_has_grad=True)
                    pred_sdf = pred_sdf
                    nablas_norm = pred_nablas.norm(dim=-1)
                    gradient_error = loss_eikonal_fn(nablas_norm)
                    loss = F.smooth_l1_loss(pred_sdf, sdf_gt, reduction='mean') + w_eikonal * gradient_error

                    # # # 2. neus
                    # sdf_nn_output= self.nerf.forward_sdf(samples_in_net)
                    # pred_sdf = sdf_nn_output[:, :1]
                    # gradient = self.nerf.gradient(samples_in_net, 0.005).squeeze()
                    # # nablas_norm =  gradient / (1e-5 + torch.linalg.norm(gradient, ord=2, dim=-1,  keepdim = True))  # 这里instant-nsr 把gridient转换成了normal，做了一个归一化
                    # pts_norm = torch.linalg.norm(samples_in_net.reshape(-1, 3), ord=2, dim=-1, keepdim=True).reshape(-1)
                    # inside_sphere = (pts_norm < 1.0).float().detach()
                    # relax_inside_sphere = (pts_norm < 1.5).float().detach()
                    # gradient_error = (torch.linalg.norm(gradient.reshape(num_points, 3), ord=2, dim=-1) - 1.0) ** 2
                    # # gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
                    # gradient_error = gradient_error.sum()
                    # loss = F.smooth_l1_loss(pred_sdf, sdf_gt, reduction='mean') + w_eikonal * gradient_error
                    
                    self.writer.add_scalar('sdf_initial/sdf_loss', F.smooth_l1_loss(pred_sdf, sdf_gt, reduction='mean'), it)
                    self.writer.add_scalar('sdf_initial/gradient_error', gradient_error, it)


                    
                    optimizer.zero_grad()
                    
                    # loss.backward()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if clip_grad_val > 0:
                        torch.nn.utils.clip_grad.clip_grad_value_(sdf_parameters, clip_grad_val)
                    
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                    

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

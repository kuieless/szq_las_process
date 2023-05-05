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

from gp_nerf.datasets.filesystem_dataset import FilesystemDataset
from gp_nerf.datasets.memory_dataset import MemoryDataset
from gp_nerf.datasets.memory_dataset_sam import MemoryDataset_SAM

from gp_nerf.image_metadata import ImageMetadata
from mega_nerf.metrics import psnr, ssim, lpips
from mega_nerf.misc_utils import main_print, main_tqdm
from mega_nerf.ray_utils import get_rays, get_ray_directions
from gp_nerf.models.model_utils import get_nerf, get_bg_nerf

import wandb
from torchvision.utils import make_grid

#semantic
from tools.unetformer.uavid2rgb import uavid2rgb, custom2rgb
from tools.unetformer.metric import Evaluator

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            # print(s)
            nn = nn*s
        pp += nn
    return pp


class Runner:
    def __init__(self, hparams: Namespace, set_experiment_path: bool = True):
        faulthandler.register(signal.SIGUSR1)

        if hparams.enable_semantic:
            CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=hparams.ignore_index)
            self.crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label)
            self.logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)

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

        else:
            self.sphere_center = None
            self.sphere_radius = None

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
            for p_base in self.nerf.encoder.parameters():
                p_base.requires_grad = False
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
                
            non_frozen_parameters = [p for p in self.nerf.parameters() if p.requires_grad]
            optimizers = {}
            optimizers['nerf'] = Adam(non_frozen_parameters, lr=self.hparams.lr)
        else:
            optimizers = {}
            optimizers['nerf'] = Adam(self.nerf.parameters(), lr=self.hparams.lr)
        
        if self.bg_nerf is not None:
            optimizers['bg_nerf'] = Adam(self.bg_nerf.parameters(), lr=self.hparams.lr)

        if self.hparams.ckpt_path is not None:
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            # # add by zyq : load the pretrain-gpnerf to train the semantic
            # if self.hparams.resume_ckpt_state:
            if False:
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
                                        self.hparams.train_scale_factor, self.hparams.disk_flush_size,self.hparams.desired_chunks)
            if self.hparams.ckpt_path is not None and ((self.hparams.resume_ckpt_state and not self.hparams.debug) and (not self.hparams.freeze_geo)):
                dataset.set_state(checkpoint['dataset_state'])
            if 'RANK' in os.environ and self.is_local_master:
                dist.barrier()
        elif self.hparams.dataset_type == 'memory':
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device)
        elif self.hparams.dataset_type == 'sam':
            dataset = MemoryDataset_SAM(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
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
                if self.hparams.dataset_type == 'sam':
                    data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0,
                                                pin_memory=False)
                else:
                    data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4,
                                                pin_memory=False)
                
            
            for dataset_index, item in enumerate(data_loader): #, start=10462):
                # item = np.load('79972.npy',allow_pickle=True).item()

                if dataset_index <= discard_index:
                    continue
                discard_index = -1

                # amp: Automatic mixed precision
                with torch.cuda.amp.autocast(enabled=self.hparams.amp):
                    if self.hparams.appearance_dim > 0:
                        image_indices = item['img_indices'].to(self.device, non_blocking=True)
                    else:
                        image_indices = None

                    #semantic 
                    if self.hparams.enable_semantic:
                        labels = item['labels'].to(self.device, non_blocking=True)
                    else:
                        labels = None

                    if self.hparams.dataset_type == 'sam':
                        groups = item['groups'].to(self.device, non_blocking=True)
                        metrics, bg_nerf_rays_present = self._training_step(
                        item['rgbs'].squeeze(0).to(self.device, non_blocking=True),
                        item['rays'].squeeze(0).to(self.device, non_blocking=True),
                        image_indices.squeeze(0), labels.squeeze(0), groups.squeeze(0), train_iterations)
                    else:
                        groups = None

                        metrics, bg_nerf_rays_present = self._training_step(
                            item['rgbs'].to(self.device, non_blocking=True),
                            item['rays'].to(self.device, non_blocking=True),
                            image_indices, labels, groups, train_iterations)

                    with torch.no_grad():
                        for key, val in metrics.items():
                            if key == 'psnr' and math.isinf(val):  # a perfect reproduction will give PSNR = infinity
                                continue

                            if not math.isfinite(val):
                                np.save(f"{train_iterations}.npy", item)
                                raise Exception('Train metrics not finite: {}'.format(metrics))


                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)

                scaler.scale(metrics['loss']).backward()
                if self.hparams.clip_grad_max != 0:
                    torch.nn.utils.clip_grad_norm_(self.nerf.parameters(), self.hparams.clip_grad_max)

                for key, optimizer in optimizers.items():
                    if key == 'bg_nerf' and (not bg_nerf_rays_present):
                        continue
                    else:
                        lr_temp = optimizer.param_groups[0]['lr']
                        if self.wandb is not None and train_iterations % self.hparams.logger_interval == 0:
                            self.wandb.log({"train/optimizer_{}_lr".format(key): lr_temp, 'epoch':train_iterations})
                        scaler.step(optimizer)

                scaler.update()
                for scheduler in schedulers.values():
                    scheduler.step()

                train_iterations += 1
                if self.is_master:
                    pbar.update(1)
                    if train_iterations % self.hparams.logger_interval ==0:
                        for key, value in metrics.items():
                            if self.writer is not None:
                                self.writer.add_scalar('train/{}'.format(key), value, train_iterations)
                            if self.wandb is not None:
                                self.wandb.log({"train/{}".format(key): value, 'epoch':train_iterations})

                    if train_iterations > 0 and train_iterations % self.hparams.ckpt_interval == 0:
                        self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index,
                                              dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)
            
                if (train_iterations > 0 and train_iterations % self.hparams.val_interval == 0) or train_iterations == self.hparams.train_iterations:
                    val_metrics = self._run_validation(train_iterations)
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
                            self.writer.add_scalar('val/psnr_avg', avg_val, train_iterations)

                    message = 'Average {}: {}'.format(key, avg_val)
                    main_print(message)
                    f.write('{}\n'.format(message))

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
            

    def _training_step(self, rgbs: torch.Tensor, rays: torch.Tensor, image_indices: Optional[torch.Tensor], labels: Optional[torch.Tensor], groups: Optional[torch.Tensor], train_iterations = -1) \
            -> Tuple[Dict[str, Union[torch.Tensor, float]], bool]:
        if self.hparams.network_type == 'sdf':
            from gp_nerf.rendering_gpnerf_clean_sdf import render_rays
        else:
            from gp_nerf.rendering_gpnerf import render_rays
        
        
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
                                                    train_iterations=train_iterations
                                                    )
        
        
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        if self.hparams.network_type == 'sdf':
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

        photo_loss = F.mse_loss(results[f'rgb_{typ}'], rgbs, reduction='mean')
        metrics['photo_loss'] = photo_loss
        
        #semantic loss
        if self.hparams.enable_semantic:
            sem_logits = results[f'sem_map_{typ}']
            semantic_loss = self.crossentropy_loss(sem_logits, labels.type(torch.long))
            metrics['semantic_loss'] = semantic_loss
            metrics['loss'] = photo_loss + self.hparams.wgt_sem_loss * semantic_loss
            with torch.no_grad():
                if train_iterations % 1000 == 0:
                    sem_label = self.logits_2_label(sem_logits)
                    if self.writer is not None:
                        self.writer.add_scalar('train_sem/accuracy', sum(labels == sem_label) / labels.shape[0], train_iterations)
                        # self.writer.add_histogram("gt_labels", labels,train_iterations)
                        # self.writer.add_histogram("pred_labels", sem_label,train_iterations)
                    if self.wandb is not None:
                        self.wandb.log({'train_sem/accuracy': sum(labels == sem_label) / labels.shape[0], 'epoch': train_iterations})
        else:
            metrics['loss'] = photo_loss

        if self.hparams.network_type == 'sdf':
            # sdf 
            metrics['gradient_error'] = results[f'gradient_error_{typ}']
            metrics['curvature_error'] = results[f'curvature_error_{typ}']
            if self.hparams.gradient_error_weight_increase and train_iterations > self.hparams.train_iterations / 2:
                gradient_error_weight = 0.1
            else:
                gradient_error_weight = self.hparams.gradient_error_weight
            metrics['loss'] = metrics['loss'] + gradient_error_weight * results[f'gradient_error_{typ}'] + 0.1 * results[f'curvature_error_{typ}']
            if self.wandb is not None:
                self.wandb.log({"train/inv_s": 1.0 / results['inv_s'], 'epoch': train_iterations})
            if self.writer is not None:
                self.writer.add_scalar('train/inv_s', 1.0 / results['inv_s'], train_iterations)
            
        return metrics, bg_nerf_rays_present

    def _run_validation(self, train_index=-1) -> Dict[str, float]:
        with torch.inference_mode():
            #semantic 
            self.metrics_val = Evaluator(num_class=self.hparams.num_semantic_classes)
            if self.hparams.label_type == 'unetformer':
                CLASSES = ('Building', 'Road', 'Tree', 'LowVeg', 'Moving_Car',  'Static_Car', 'Human', 'Clutter')
            elif self.hparams.label_type == 'm2f_custom':
                CLASSES = ('Cluster', 'Building', 'Road', 'Car', 'Tree', 'Vegetation', 'Human', 'Sky', 'Water', 'Ground', 'Mountain')
            
            self.nerf.eval()
            val_metrics = defaultdict(float)
            base_tmp_path = None
            
            val_type = self.hparams.val_type  # train  val
            print('val_type: ', val_type)
            try:
                if 'RANK' in os.environ:
                    base_tmp_path = Path(self.hparams.exp_name) / os.environ['TORCHELASTIC_RUN_ID']
                    metric_path = base_tmp_path / 'tmp_val_metrics'
                    image_path = base_tmp_path / 'tmp_val_images'

                    world_size = int(os.environ['WORLD_SIZE'])
                    indices_to_eval = np.arange(int(os.environ['RANK']), len(self.val_items), world_size)
                    if self.is_master:
                        base_tmp_path.mkdir()
                        metric_path.mkdir()
                        image_path.mkdir()
                    dist.barrier()
                else:
                    if val_type == 'val':
                        if 'residence'in self.hparams.dataset_path:
                            self.val_items=self.val_items[:19]
                        elif 'building'in self.hparams.dataset_path or 'campus'in self.hparams.dataset_path:
                            self.val_items=self.val_items[:10]
                        indices_to_eval = np.arange(len(self.val_items))
                    elif val_type == 'train':
                        indices_to_eval = np.arange(200)  #np.arange(len(self.train_items))
                experiment_path_current = self.experiment_path / "eval_{}".format(train_index)
                Path(str(experiment_path_current)).mkdir()
                Path(str(experiment_path_current / 'val_rgbs')).mkdir()
                with (experiment_path_current / 'psnr.txt').open('w') as f, (experiment_path_current / 'iou.txt').open('w') as f2:
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

                        # semantic
                        if self.hparams.enable_semantic:
                            sem_logits = results[f'sem_map_{typ}']
                            
                            if val_type == 'val':
                                    gt_label = metadata_item.load_gt()
                            elif val_type == 'train':
                                gt_label = metadata_item.load_label()
                            
                            gt_label_rgb = custom2rgb(gt_label.view(*viz_rgbs.shape[:-1]).cpu().numpy())
                            sem_label = self.logits_2_label(sem_logits)
                            visualize_sem = custom2rgb(sem_label.view(*viz_rgbs.shape[:-1]).cpu().numpy())
                            gt_class = gt_label.view(-1)


                            # OA, mIoU
                            self.metrics_val.add_batch(gt_class.cpu().numpy(), sem_label.cpu().numpy())
                            self.metrics_val_each.add_batch(gt_class.cpu().numpy(), sem_label.cpu().numpy())
                        viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()
                        viz_result_rgbs = viz_result_rgbs.clamp(0,1)
                        
                        #calculate psnr  ssim  lpips ******************************:
                        if val_type == 'val':
                            eval_rgbs = viz_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
                            eval_result_rgbs = viz_result_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()

                            val_psnr = psnr(eval_result_rgbs.view(-1, 3), eval_rgbs.view(-1, 3))

                            main_print('The psnr of the {} image is: {}'.format(i, val_psnr))
                            f.write('The psnr of the {} image is: {}\n'.format(i, val_psnr))

                            metric_key = 'val/psnr/{}'.format(train_index)
                            if self.wandb is not None:
                                self.wandb.log({'val/psnr/{}'.format(train_index): val_psnr, 'epoch': i})
                            if self.writer is not None:
                                self.writer.add_scalar(metric_key, val_psnr, i)

                            val_metrics['val/psnr'] += val_psnr

                            val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1)

                            metric_key = 'val/ssim/{}'.format(train_index)

                            # TODO: 暂时不放ssim
                            if self.wandb is not None:
                                self.wandb.log({'val/ssim/{}'.format(train_index): val_ssim, 'epoch':i})
                            if self.writer is not None:
                                self.writer.add_scalar(metric_key, val_ssim, i)

                            val_metrics['val/ssim'] += val_ssim

                            val_lpips_metrics = lpips(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs)

                            for network in val_lpips_metrics:
                                agg_key = 'val/lpips/{}'.format(network)
                                metric_key = '{}/{}'.format(agg_key, train_index)
                                # TODO: 暂时不放lpips
                                # if self.wandb is not None:
                                #     self.wandb.log({'val/lpips/{}/{}'.format(network, train_index): val_lpips_metrics[network], 'epoch':i})
                                # if self.writer is not None:
                                #     self.writer.add_scalar(metric_key, val_lpips_metrics[network], i)

                                val_metrics[agg_key] += val_lpips_metrics[network]
                        #  calculate psnr  ssim  lpips ******************************:

                        viz_result_rgbs = viz_result_rgbs.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                        if f'depth_{typ}' in results:
                            viz_depth = results[f'depth_{typ}']
                            if f'fg_depth_{typ}' in results:
                                to_use = results[f'fg_depth_{typ}'].view(-1)
                                while to_use.shape[0] > 2 ** 24:
                                    to_use = to_use[::2]
                                ma = torch.quantile(to_use, 0.95)

                                viz_depth = viz_depth.clamp_max(ma)
                        else: 
                            viz_depth = None
                        
                        ################################## visualize all
                        if self.hparams.enable_semantic:
                            
                            if val_type == 'val':
                                gt_label_rgb = torch.from_numpy(gt_label_rgb)
                                pseudo_gt_label_rgb = metadata_item.load_label()
                                pseudo_gt_label_rgb = custom2rgb(pseudo_gt_label_rgb.view(*viz_rgbs.shape[:-1]).cpu().numpy())
                                pseudo_gt_label_rgb = torch.from_numpy(pseudo_gt_label_rgb)
                            elif val_type == 'train':
                                pseudo_gt_label_rgb = torch.from_numpy(gt_label_rgb)
                                gt_label_rgb = None
                            
                            # NOTE: 这里初始化了一个list，需要可视化的东西可以后续加上去
                            img_list = [viz_rgbs * 255, gt_label_rgb, pseudo_gt_label_rgb, viz_result_rgbs * 255, torch.from_numpy(visualize_sem)]
                            
                            if self.hparams.network_type == 'sdf':
                                #  NSR  SDF ------------------------------------  save the normal_map
                                # world -> camera 
                                w2c = torch.linalg.inv(torch.cat((metadata_item.c2w,torch.tensor([[0,0,0,1]])),0))
                                viz_result_normal_map = results[f'normal_map_{typ}']
                                # viz_result_normal_map = torch.mm(viz_result_normal_map, w2c[:3,:3])# + w2c[:3,3]
                                viz_result_normal_map = torch.mm(w2c[:3,:3],viz_result_normal_map.T).T
                                # normalize 
                                viz_result_normal_map = viz_result_normal_map / (1e-5 + torch.linalg.norm(viz_result_normal_map, ord = 2, dim=-1, keepdim=True))
                                # viz_result_normal_map = viz_result_normal_map.view(*viz_rgbs.shape).cpu()
                                viz_result_normal_map = viz_result_normal_map.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                                
                                # img = Runner._create_rendering_semantic(viz_rgbs, gt_label_rgb, pseudo_gt_label_rgb, 
                                #                                     viz_result_rgbs, torch.from_numpy(visualize_sem), viz_result_normal_map)
                                # img_list = Runner._create_res_list(viz_rgbs, gt_label_rgb, pseudo_gt_label_rgb, viz_result_rgbs, torch.from_numpy(visualize_sem), viz_result_normal_map)
                                
                                normal_viz = (viz_result_normal_map+1)*0.5*255
                                img_list.append(normal_viz)
                            else:
                                if viz_depth is not None:
                                    depth_vis = torch.from_numpy(Runner.visualize_scalars(torch.log(viz_depth + 1e-8).view(viz_rgbs.shape[0], viz_rgbs.shape[1]).cpu()))
                                img_list.append(depth_vis)
                            
                            # NOTE: 对需要可视化的list进行处理
                            # list：  N * (H, W, 3),  -> tensor(N, 3, H, W)
                            # 将None元素转换为zeros矩阵
                            img_list = [torch.zeros_like(viz_rgbs) if element is None else element for element in img_list]
                            img_list = torch.stack(img_list).permute(0,3,1,2)
                            img = make_grid(img_list, nrow=3)
                            img_grid = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                            Image.fromarray(img_grid).save(str(experiment_path_current / 'val_rgbs' / '{}_all.jpg'.format(i)))
                            
                            # img.save(str(experiment_path_current / 'val_rgbs' / '{}_all.jpg'.format(i)))
                            
                            if self.writer is not None:
                                self.writer.add_image('val/{}'.format(i), img.byte(), train_index)
                            if self.wandb is not None:
                                Img = wandb.Image(img, caption="ckpt {}: {} th".format(train_index, i))
                                self.wandb.log({"images_all/{}".format(train_index): Img, 'epoch': i})
                        
                        # NOTE： 这里的else是原始的GPNeRF可视化
                        else:
                            img = Runner._create_result_image(viz_rgbs, viz_result_rgbs, viz_depth)
                            if self.wandb is not None:
                                Img = wandb.Image(T.ToTensor()(img), caption="ckpt {}: {} th".format(train_index, i))
                                self.wandb.log({"images_val/{}".format(train_index): Img})
                            if self.writer is not None:
                                self.writer.add_image('val/{}'.format(i), T.ToTensor()(img), train_index)
                            img.save(str(experiment_path_current / 'val_rgbs' / '{}.jpg'.format(i)))

                        if val_type == 'val':
                            ##################################   pred   label & rgb
                            # if self.wandb is not None:
                            #     Img = wandb.Image((viz_result_rgbs.numpy() * 255).astype(np.uint8),
                            #                       caption="ckpt {}: {} th; psnr is {}".format(train_index, i, val_psnr))
                            #     self.wandb.log({"images_pred_rgbs/{}".format(train_index): Img})
                            Image.fromarray((viz_result_rgbs.numpy() * 255).astype(np.uint8)).save(
                                str(experiment_path_current / 'val_rgbs' / '{}_pred_rgb.jpg'.format(i)))
                            if self.hparams.enable_semantic:
                                Image.fromarray((visualize_sem).astype(np.uint8)).save(
                                    str(experiment_path_current / 'val_rgbs' / '{}_pred_label.jpg'.format(i)))

                            ##################################   fg & bg
                            if self.hparams.bg_nerf or f'bg_rgb_{typ}' in results:
                                img = Runner._create_fg_bg_image(results[f'fg_rgb_{typ}'].view(viz_rgbs.shape[0],viz_rgbs.shape[1], 3).cpu(),
                                                                 results[f'bg_rgb_{typ}'].view(viz_rgbs.shape[0],viz_rgbs.shape[1], 3).cpu())
                                img.save(str(experiment_path_current / 'val_rgbs' / '{}_fg_bg.jpg'.format(i)))
                            
                                # if self.wandb is not None:
                                #     Img = wandb.Image(T.ToTensor()(img), caption="ckpt {}: {} th".format(train_index, i))
                                #     self.wandb.log({"images_fg_bg/{}".format(train_index): Img})
                                # if self.writer is not None:
                                #     self.writer.add_image('val/{}_fg_bg'.format(i), T.ToTensor()(img), train_index)
                            ################################## 

                                            # OA, mIoU
                                mIoU = np.nanmean(self.metrics_val_each.Intersection_over_Union())
                                F1 = np.nanmean(self.metrics_val_each.F1())
                                OA = np.nanmean(self.metrics_val_each.OA())
                                iou_per_class = self.metrics_val_each.Intersection_over_Union()
                                eval_value = {'mIoU': mIoU,
                                            'F1': F1,
                                            'OA': OA}
                                iou_value = {}
                                for class_name, iou in zip(CLASSES, iou_per_class):
                                    iou_value[class_name] = iou
                                f2.write('*'*10+f'{i}'+'*'*10+'\n')
                                f2.write('eval_value:\n')
                                for key in eval_value:
                                    f2.write(f'{key:<12}: {eval_value[key]}\n')
                                f2.write('iou_value:\n')
                                for key in iou_value:
                                    f2.write(f'{key:<12}: {iou_value[key]}\n' )
                                f2.write('\n')

                                if self.wandb is not None:
                                    self.wandb.log({'val/mIoU_each/{}'.format(train_index): mIoU, 'epoch':i})
                                    self.wandb.log({'val/F1_each/{}'.format(train_index): F1, 'epoch':i})
                                    self.wandb.log({'val/OA_each/{}'.format(train_index): OA, 'epoch':i})
                                if self.writer is not None:
                                    self.writer.add_scalar('val/mIoU_each/{}'.format(train_index), mIoU, i)
                                    self.writer.add_scalar('val/F1_each/{}'.format(train_index), F1, i)
                                    self.writer.add_scalar('val/OA_each/{}'.format(train_index), OA, i)
                                self.metrics_val_each.reset()
                        del results

                # OA, mIoU
                mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
                F1 = np.nanmean(self.metrics_val.F1())
                OA = np.nanmean(self.metrics_val.OA())
                iou_per_class = self.metrics_val.Intersection_over_Union()
                print("eval_value")
                eval_value = {'mIoU': mIoU,
                              'F1': F1,
                              'OA': OA}
                print('val:', eval_value)
                iou_value = {}
                for class_name, iou in zip(CLASSES, iou_per_class):
                    iou_value[class_name] = iou
                print(iou_value)
                experiment_path_current = self.experiment_path / "eval_{}".format(train_index)
                with (experiment_path_current /'metrics.txt').open('a') as f:
                    f.write('eval_value:\n')
                    for key in eval_value:
                        f.write(f'{key:<12}: {eval_value[key]}\n')
                    f.write('iou_value:\n')
                    for key in iou_value:
                        f.write(f'{key:<12}: {iou_value[key]}\n' )
                        # f.write('eval_value:\n{}\niou_value:\n{}\n'.format(eval_value, iou_value))
                
                if self.wandb is not None:
                    self.wandb.log({'val/mIoU': mIoU, 'epoch':train_index})
                    self.wandb.log({'val/F1': F1, 'epoch':train_index})
                    self.wandb.log({'val/OA': OA, 'epoch':train_index})
                if self.writer is not None:
                    self.writer.add_scalar('val/mIoU', mIoU, train_index)
                    self.writer.add_scalar('val/F1', F1, train_index)
                    self.writer.add_scalar('val/OA', OA, train_index)

                self.writer.flush()
                self.writer.close()

                self.metrics_val.reset()

                if 'RANK' in os.environ:
                    dist.barrier()
                    if self.writer is not None:
                        for metric_file in metric_path.iterdir():
                            metric = torch.load(metric_file, map_location='cpu')
                            self.writer.add_scalar(metric['metric_key'], metric['value'], train_index)


                            val_metrics[metric['agg_key']] += metric['value']
                        for image_file in image_path.iterdir():
                            img = Image.open(str(image_file))
                            self.writer.add_image('val/{}'.format(image_file.stem), T.ToTensor()(img), train_index)
                        for key in val_metrics:
                            avg_val = val_metrics[key] / len(self.val_items)
                            self.writer.add_scalar('{}/avg'.format(key), avg_val, 0)



                    dist.barrier()

                self.nerf.train()
            finally:
                if self.is_master and base_tmp_path is not None:
                    shutil.rmtree(base_tmp_path)

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
        if self.hparams.network_type == 'sdf':
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

        with torch.cuda.amp.autocast(enabled=self.hparams.amp):
            rays = get_rays(directions, metadata.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)

            rays = rays.view(-1, 8).to(self.device, non_blocking=True).cuda()  # (H*W, 8)
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
                                              train_iterations=train_index)

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
    def visualize_scalars(scalar_tensor: torch.Tensor) -> np.ndarray:
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
        if self.hparams.dataset_type == 'sam':
            # train_paths=train_paths[:10]
            # val_paths = val_paths[:2]
            image_indices_path = train_paths + val_paths
            image_indices_path.sort(key=lambda x: x.name)
            val_paths_set = set(val_paths)
            image_indices = {}
            for i, indices_path in enumerate(image_indices_path):
                image_indices[indices_path.name] = i
        else:
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

        assert image_path.exists()

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


    
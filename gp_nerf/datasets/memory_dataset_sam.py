from typing import List, Dict

import torch
from torch.utils.data import Dataset

from gp_nerf.datasets.dataset_utils import get_rgb_index_mask_sam
from gp_nerf.image_metadata import ImageMetadata
from mega_nerf.misc_utils import main_tqdm, main_print
from mega_nerf.ray_utils import get_rays, get_ray_directions
from tools.segment_anything import sam_model_registry, SamPredictor

import numpy as np
import cv2
import sys
import glob
import os
from pathlib import Path
import random
import shutil

from tools.unetformer.uavid2rgb import remapping, custom2rgb


def init(device):
    # sam_checkpoint = "samnerf/segment_anything/sam_vit_h_4b8939.pth"
    sam_checkpoint = "tools/segment_anything/sam_vit_h_4b8939.pth"
    
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


class MemoryDataset_SAM(Dataset):

    def __init__(self, metadata_items: List[ImageMetadata], near: float, far: float, ray_altitude_range: List[float],
                 center_pixels: bool, device: torch.device, hparams):
        super(MemoryDataset_SAM, self).__init__()

        self.debug_one_images_sam = hparams.debug_one_images_sam
        print(f"train_on_1_val_corresponding_images: {hparams.debug_one_images_sam}")

        self.online_sam_label = hparams.online_sam_label
        self.visualization = False
        self.Visual_num = 0
        self.Visual_Num = 5
        # sam
        self.device = device # device 'cpu'
        self.predictor = init(self.device)
        self.N_total = hparams.sample_ray_num
        self.N_each = hparams.sam_sample_each
        # 假设所有图像的长宽一致
        self.W = metadata_items[0].W
        self.H = metadata_items[0].H
        self.sampling_mode = hparams.sampling_mode
        #get ray
        self.near = near
        self.far = far
        self.metadata_items = metadata_items
        self.ray_altitude_range = ray_altitude_range
        metadata_item=metadata_items[0]
        self.directions = get_ray_directions(metadata_item.W,
                                            metadata_item.H,
                                            metadata_item.intrinsics[0],
                                            metadata_item.intrinsics[1],
                                            metadata_item.intrinsics[2],
                                            metadata_item.intrinsics[3],
                                            center_pixels,
                                            device)

        # rgbs = []
        rays = []
        indices = []
        labels = []
        sam_features = []
        is_vals =[]

        main_print('Loading data')
        if hparams.debug:
            metadata_items = metadata_items[:40]
        
        if self.debug_one_images_sam:
            used_files = []
            used_files.extend(glob.glob(os.path.join('/data/yuqi/code/GP-NeRF-semantic/zyq/val_000459_v4', '*.jpg')))
            used_files.extend(glob.glob(os.path.join('/data/yuqi/code/GP-NeRF-semantic/zyq/val_000459_v4/occluded', '*.jpg')))
            used_files.sort()
            use_index = []
            for used_file in used_files:
                use_index.append(int(Path(used_file).stem))
            use_index.sort()
            self.use_index = use_index
        
        for metadata_item in main_tqdm(metadata_items):
            image_data = get_rgb_index_mask_sam(metadata_item)

            if image_data is None:
                continue
            
            #zyq : add labels
            image_rgbs, image_indices, image_keep_mask, label, sam_feature, is_val = image_data

            indices.append(image_indices)
            labels.append(torch.tensor(label, dtype=torch.int))
            sam_features.append(sam_feature)
            is_vals.append(is_val)
        
        main_print('Finished loading data')

        self._img_indices = indices  #torch.stack(indices)
        self._labels = torch.stack(labels)
        self._sam_features = torch.stack(sam_features)
        self._is_vals = is_vals



    def __len__(self) -> int:
        # return self._rgbs.shape[0]
        return self._labels.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if self.debug_one_images_sam:
            if not (idx in self.use_index):
                random_number = random.randint(0, len(self.use_index))
                idx = random_number
        
        if self._is_vals[idx]:
            idx = idx + 1
            assert self._is_vals[idx] == False
        
        in_labels = np.array([1])
        H, W = self.H, self.W
        sam_feature = (self._sam_features[idx]).to(self.device)
        self.predictor.set_feature(sam_feature, [H, W])
        N_selected = 0
        selected_points = []
        selected_points_labels = []
        selected_points_group = []
        group_id = 1  #  0 denotes unassigned
        selected_one = []
        #visualize
        visual = torch.zeros((H, W), dtype=torch.int).to(self.device) 
        bool_tensor = torch.ones((H, W), dtype=torch.int).to(self.device)
        
        if self.visualization and self.Visual_num < self.Visual_Num:
            save_dir = f'zyq/visual_sam_sample_{idx}'
            Path(save_dir).parent.mkdir(exist_ok=True)
            Path(save_dir).mkdir(exist_ok=True)



        visual_rgb = self.metadata_items[idx].load_image()
        visual_rgb = np.flip(visual_rgb.numpy(), axis=2)
        visual_pseudo = np.flip(custom2rgb(remapping(self._labels[idx].numpy())), axis=2)
        
        # mode: 2  先随机采样所有点，然后从采样点中取一点，识别mask，划分group； 再拿未识别的点，重复上述过程
        # 还有问题，到后面很慢
        if self.sampling_mode == 'whole_image':
            ignore_point_index=[]
            random_index = torch.randperm(H*W)[:self.N_total].to(self.device) 
            rows = torch.div(random_index, W, rounding_mode='trunc')
            cols = random_index % W
            selected_points = torch.stack([rows, cols],dim=1)

            selected_points_status = torch.zeros((H*W), dtype=torch.int).to(self.device)
            selected_points_status[random_index] = 1
            selected_points_labels = -1 * torch.ones_like(random_index)
            selected_points_groups = -1 * torch.ones_like(random_index)

            while selected_points_status.nonzero().size(0) > 0:  # bool_tensor 为空， 说明分配完了
                random_point = torch.nonzero(selected_points_status.view(H, W))[torch.randint(high=torch.sum(selected_points_status), size=(1,))]

                random_point = random_point.flip(1)
                selected_one.append(random_point)
                masks, iou_preds, _ = self.predictor.predict(random_point.cpu().numpy(), in_labels, return_torch=True)
                masks_max = masks[:, torch.argmax(iou_preds),:,:].squeeze(0)
                masks_max = masks_max * bool_tensor.bool()
                mask_size = masks_max.nonzero().size(0)

                if mask_size == 0:
                    continue

                if self.online_sam_label:
                    sam_cut_out_labels = self._labels[idx][masks_max==True]
                    cut_out_labels_uniques, cut_out_labels_counts = torch.unique(sam_cut_out_labels, return_counts=True)
                    max_label = cut_out_labels_uniques[cut_out_labels_counts.argmax()].int()
                    cut_out_labels_nonzero = cut_out_labels_uniques.nonzero().squeeze(-1)
                    if max_label == 0 and len(cut_out_labels_nonzero)>0:
                        cut_out_labels_uniques_nonzero = cut_out_labels_uniques[cut_out_labels_nonzero]
                        cut_out_labels_counts_nonzero = cut_out_labels_counts[cut_out_labels_nonzero]
                        max_label_nonzero = cut_out_labels_uniques_nonzero[cut_out_labels_counts_nonzero.argmax()]
                        max_count_nonzero = cut_out_labels_counts_nonzero[cut_out_labels_counts_nonzero.argmax()]
                        if max_count_nonzero / mask_size >0.1:
                            max_label=max_label_nonzero.int()


                current_index_flatten = masks_max.view(-1)

                #种子点没有落在mask里面
                random_point_index = random_point[0,1]*W+random_point[0,0]
                if current_index_flatten[random_point_index]==False:
                    ignore_point_index.append(torch.where(random_index == random_point_index))
                    selected_points_status[random_point_index]= 0

                selected_points_status[current_index_flatten] = 0
                
                #寻找落在当前mask中的采样点
                current_index = current_index_flatten[random_index]
                selected_points_labels[current_index] = max_label
                group_id_record = 1000 * idx + group_id
                selected_points_groups[current_index]=group_id_record
                group_id += 1
                bool_tensor = bool_tensor * (~masks_max)

                if self.visualization and self.Visual_num < self.Visual_Num:
                    N_selected += current_index.nonzero().size(0)
                    print(N_selected)
                    visual += masks_max * max_label
                    masks_max_vis = np.flip(custom2rgb(remapping(visual.cpu().numpy())),axis=2)
                    for point in selected_one:
                        # print(point)
                        for k in range(point.size(0)):  # W , H
                            x, y = int(point[k, 0]), int(point[k, 1])
                            for m in [0]: #range(-5,5):
                                for n in [0]: #range(-5,5):
                                    if x+m < W and x+m >0 and y+n < H and y+n >0:
                                        masks_max_vis[(y+n), (x+m)] = np.array([14, 7, 176])
                    if os.path.exists(f"{save_dir}/%06d_mask_%06d.png" % (idx, N_selected)):
                        cv2.imwrite(f"{save_dir}/%06d_mask_%06d_2.png" % (idx, N_selected), np.concatenate([visual_rgb, visual_pseudo, masks_max_vis], axis=1))
                    else:
                        cv2.imwrite(f"{save_dir}/%06d_mask_%06d.png" % (idx, N_selected), np.concatenate([visual_rgb, visual_pseudo, masks_max_vis], axis=1))

            assert (selected_points_labels==-1).nonzero() == 0
            assert (selected_points_groups==-1).nonzero() == 0
                
            if self.visualization and self.Visual_num < self.Visual_Num:
                visual = visual.cpu().numpy()
                rgb_array = np.flip(custom2rgb(remapping(visual)),axis=2)

                for point in selected_points:
                    for k in range(point.size(0)):  # H, W
                        x, y = point[k, 0], point[k, 1]
                        if x < H and x >0 and y < W and y >0:
                            rgb_array[x, y] = np.array([0, 128, 0])
                for point in selected_one:
                    # print(point)
                    for k in range(point.size(0)):  # W , H
                        x, y = int(point[k, 0]), int(point[k, 1])
                        for m in range(-5,5):
                            for n in range(-5,5):
                                if x+m < W and x+m >0 and y+n < H and y+n >0:
                                    rgb_array[(y+n), (x+m)] = np.array([0, 0, 128])
                image_cat = np.concatenate([visual_rgb, visual_pseudo, rgb_array], axis=1)
                cv2.imwrite(f"{save_dir}/%06d_mask_all.png" % idx, image_cat)
                print(f"{save_dir}/%06d_mask_all.png" % idx)
                self.Visual_num += 1
            # *****************************************************************************
            
            selected_points = torch.index_select(selected_points, 0, torch.tensor([i for i in range(selected_points.size(0)) if i not in ignore_point_index]))
            selected_points_labels = torch.index_select(selected_points_labels, 0, torch.tensor([i for i in range(selected_points_labels.size(0)) if i not in ignore_point_index]))
            selected_points_groups = torch.index_select(selected_points_groups, 0, torch.tensor([i for i in range(selected_points_groups.size(0)) if i not in ignore_point_index]))

        elif self.sampling_mode == 'per_mask_threshold':
            while N_selected < self.N_total:
                random_point = torch.nonzero(bool_tensor)[torch.randint(high=torch.sum(bool_tensor), size=(1,))]
                random_point = random_point.flip(1)
                selected_one.append(random_point)
                masks, iou_preds, _ = self.predictor.predict(random_point.cpu().numpy(), in_labels, return_torch=True)
                masks_max = masks[:, torch.argmax(iou_preds),:,:].squeeze(0)
                masks_max = masks_max * bool_tensor.bool()
                images_size = H*W
                mask_size = masks_max.nonzero().size(0)
                if masks_max.nonzero().size(0) == 0:
                    continue
                if self.online_sam_label:
                    sam_cut_out_labels = self._labels[idx][masks_max==True]
                    cut_out_labels_uniques, cut_out_labels_counts = torch.unique(sam_cut_out_labels, return_counts=True)
                    max_label = cut_out_labels_uniques[cut_out_labels_counts.argmax()].int()
                    cut_out_labels_nonzero = cut_out_labels_uniques.nonzero().squeeze(-1)
                    if max_label == 0 and len(cut_out_labels_nonzero)>0:
                        cut_out_labels_uniques_nonzero = cut_out_labels_uniques[cut_out_labels_nonzero]
                        cut_out_labels_counts_nonzero = cut_out_labels_counts[cut_out_labels_nonzero]
                        max_label_nonzero = cut_out_labels_uniques_nonzero[cut_out_labels_counts_nonzero.argmax()]
                        max_count_nonzero = cut_out_labels_counts_nonzero[cut_out_labels_counts_nonzero.argmax()]
                        if max_count_nonzero / mask_size >0.1:
                            max_label=max_label_nonzero.int()

                mask_ratio = mask_size / images_size
                if mask_ratio < 0.02:
                    N_sample = min(int(self.N_total*0.01), mask_size)
                elif mask_ratio < 0.05:
                    N_sample = min(int(self.N_total*0.05), mask_size)
                elif mask_ratio > 0.8:
                    N_sample = min(int(self.N_total), mask_size)
                elif mask_ratio > 0.5:
                    N_sample = min(int(self.N_total*0.6), mask_size)
                elif mask_ratio > 0.3:
                    N_sample = min(int(self.N_total*0.3), mask_size)
                else:
                    N_sample = min(int(self.N_total*0.2), mask_size)
                select_point = torch.nonzero(masks_max)[torch.randint(high=torch.sum(masks_max), size=(N_sample,))]
                candidate_tensor = torch.ones(select_point.size(0),dtype=torch.int).to(self.device)
                selected_points.append(select_point)
                group_id_record = 1000 * idx + group_id
                selected_points_group.append(group_id_record * candidate_tensor)
                N_selected += N_sample
                
                if self.online_sam_label:
                    selected_points_labels.append(max_label * candidate_tensor)


                group_id += 1
                bool_tensor = bool_tensor * (~masks_max)
                
                # *****************************************************************************
                if self.visualization and self.Visual_num < self.Visual_Num:
                    masks_max_vis = np.stack([masks_max.cpu().numpy(), masks_max.cpu().numpy(), masks_max.cpu().numpy()], axis=2)
                    for point in selected_one:
                        # print(point)
                        for k in range(point.size(0)):  # W , H
                            x, y = int(point[k, 0]), int(point[k, 1])
                            for m in range(-2,2):
                                for n in range(-2,2):
                                    if x+m < W and x+m >0 and y+n < H and y+n >0:
                                        masks_max_vis[(y+n), (x+m)] = np.array([0, 0, 128])
                    visual += masks_max * max_label
                    masks_max_vis = np.flip(custom2rgb(remapping(visual.cpu().numpy())),axis=2)
                    # cv2.imwrite(f"{save_dir}/%06d_mask_%06d.png" % (idx, N_selected), np.concatenate([visual_rgb, visual_pseudo, masks_max_vis], axis=1))
            if self.visualization and self.Visual_num < self.Visual_Num:
                visual = visual.cpu().numpy()
                rgb_array = np.flip(custom2rgb(remapping(visual)),axis=2)

                for point in selected_points:
                    for k in range(point.size(0)):  # H, W
                        x, y = point[k, 0], point[k, 1]
                        if x < H and x >0 and y < W and y >0:
                            rgb_array[x, y] = np.array([0, 128, 0])
                for point in selected_one:
                    # print(point)
                    for k in range(point.size(0)):  # W , H
                        x, y = int(point[k, 0]), int(point[k, 1])
                        for m in range(-5,5):
                            for n in range(-5,5):
                                if x+m < W and x+m >0 and y+n < H and y+n >0:
                                    rgb_array[(y+n), (x+m)] = np.array([0, 0, 128])
                image_cat = np.concatenate([visual_rgb, visual_pseudo, rgb_array], axis=1)
                cv2.imwrite(f"{save_dir}/%06d_mask_all.png" % idx, image_cat)
                print(f"{save_dir}/%06d_mask_all.png" % idx)
                self.Visual_num += 1
            # *****************************************************************************

            selected_points = torch.cat(selected_points)
            selected_points_group = torch.cat(selected_points_group)
            if self.online_sam_label:
                selected_points_labels = torch.cat(selected_points_labels)
            
            assert selected_points.size(0) == selected_points_group.size(0)
            if selected_points.size(0) > self.N_total:
                selected_points = selected_points[:self.N_total]
                selected_points_group = selected_points_group[:self.N_total]
                if self.online_sam_label:
                    selected_points_labels = selected_points_labels[:self.N_total]
        
        
        elif self.sampling_mode == 'per_mask_threshold_old_0512':
            while N_selected < self.N_total:
                random_point = torch.nonzero(bool_tensor)[torch.randint(high=torch.sum(bool_tensor), size=(1,))]
                random_point = random_point.flip(1)
                selected_one.append(random_point)
                masks, iou_preds, _ = self.predictor.predict(random_point.cpu().numpy(), in_labels, return_torch=True)
                masks_max = masks[:, torch.argmax(iou_preds),:,:].squeeze(0)
                masks_max = masks_max * bool_tensor.bool()
                images_size = H*W
                mask_size = masks_max.nonzero().size(0)
                if masks_max.nonzero().size(0) == 0:
                    continue
                if self.online_sam_label:
                    sam_cut_out_labels = self._labels[idx][masks_max==True]
                    cut_out_labels_uniques, cut_out_labels_counts = torch.unique(sam_cut_out_labels, return_counts=True)
                    max_label = cut_out_labels_uniques[cut_out_labels_counts.argmax()].int()
                    cut_out_labels_nonzero = cut_out_labels_uniques.nonzero().squeeze(-1)
                    if max_label == 0 and len(cut_out_labels_nonzero)>0:
                        cut_out_labels_uniques_nonzero = cut_out_labels_uniques[cut_out_labels_nonzero]
                        cut_out_labels_counts_nonzero = cut_out_labels_counts[cut_out_labels_nonzero]
                        max_label_nonzero = cut_out_labels_uniques_nonzero[cut_out_labels_counts_nonzero.argmax()]
                        max_count_nonzero = cut_out_labels_counts_nonzero[cut_out_labels_counts_nonzero.argmax()]
                        if max_count_nonzero / mask_size >0.1:
                            max_label=max_label_nonzero.int()

                if mask_size / images_size < 0.02:
                    N_sample = min(int(self.N_each*0.1), mask_size)
                elif mask_size / images_size < 0.05:
                    N_sample = min(int(self.N_each*0.2), mask_size)
                else:
                    N_sample = min(int(self.N_each*1), mask_size)
                select_point = torch.nonzero(masks_max)[torch.randint(high=torch.sum(masks_max), size=(N_sample,))]
                candidate_tensor = torch.ones(select_point.size(0),dtype=torch.int).to(self.device)
                selected_points.append(select_point)
                group_id_record = 1000 * idx + group_id
                selected_points_group.append(group_id_record * candidate_tensor)
                N_selected += N_sample
                
                if self.online_sam_label:
                    selected_points_labels.append(max_label * candidate_tensor)


                group_id += 1
                bool_tensor = bool_tensor * (~masks_max)
                
                # *****************************************************************************
                if self.visualization and self.Visual_num < self.Visual_Num:
                    masks_max_vis = np.stack([masks_max.cpu().numpy(), masks_max.cpu().numpy(), masks_max.cpu().numpy()], axis=2)
                    for point in selected_one:
                        # print(point)
                        for k in range(point.size(0)):  # W , H
                            x, y = int(point[k, 0]), int(point[k, 1])
                            for m in range(-2,2):
                                for n in range(-2,2):
                                    if x+m < W and x+m >0 and y+n < H and y+n >0:
                                        masks_max_vis[(y+n), (x+m)] = np.array([0, 0, 128])
                    visual += masks_max * max_label
                    masks_max_vis = np.flip(custom2rgb(remapping(visual.cpu().numpy())),axis=2)
                    cv2.imwrite(f"{save_dir}/%06d_mask_%06d.png" % (idx, N_selected), np.concatenate([visual_rgb, visual_pseudo, masks_max_vis], axis=1))
            if self.visualization and self.Visual_num < self.Visual_Num:
                visual = visual.cpu().numpy()
                rgb_array = np.flip(custom2rgb(remapping(visual)),axis=2)

                for point in selected_points:
                    for k in range(point.size(0)):  # H, W
                        x, y = point[k, 0], point[k, 1]
                        if x < H and x >0 and y < W and y >0:
                            rgb_array[x, y] = np.array([0, 128, 0])
                for point in selected_one:
                    # print(point)
                    for k in range(point.size(0)):  # W , H
                        x, y = int(point[k, 0]), int(point[k, 1])
                        for m in range(-5,5):
                            for n in range(-5,5):
                                if x+m < W and x+m >0 and y+n < H and y+n >0:
                                    rgb_array[(y+n), (x+m)] = np.array([0, 0, 128])
                image_cat = np.concatenate([visual_rgb, visual_pseudo, rgb_array], axis=1)
                cv2.imwrite(f"{save_dir}/%06d_mask_all.png" % idx, image_cat)
                print(f"{save_dir}/%06d_mask_all.png" % idx)
                self.Visual_num += 1
            # *****************************************************************************

            selected_points = torch.cat(selected_points)
            selected_points_group = torch.cat(selected_points_group)
            if self.online_sam_label:
                selected_points_labels = torch.cat(selected_points_labels)
            
            assert selected_points.size(0) == selected_points_group.size(0)
            if selected_points.size(0) > self.N_total:
                selected_points = selected_points[:self.N_total]
                selected_points_group = selected_points_group[:self.N_total]
                if self.online_sam_label:
                    selected_points_labels = selected_points_labels[:self.N_total]
        
        else:
            raise ImportError
        
        metadata_item = self.metadata_items[idx]
        image_rays = get_rays(self.directions, metadata_item.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)
        img_indices = self._img_indices[idx] * torch.ones(selected_points.shape[0], dtype=torch.int32)
        
        if self.online_sam_label:
            item = {
            # 'rgbs': self._rgbs[idx, selected_points[:, 0], selected_points[:, 1], :],
            'rays': image_rays[selected_points[:, 0], selected_points[:, 1], :], 
            'img_indices': img_indices,
            'labels': selected_points_labels,
            'groups': selected_points_group,
            # 'selected_points': selected_points
            }
        else:
            item = {
                # 'rgbs': self._rgbs[idx, selected_points[:, 0], selected_points[:, 1], :],
                'rays': image_rays[selected_points[:, 0], selected_points[:, 1], :], 
                'img_indices': img_indices,
                'labels': self._labels[idx, selected_points[:, 0], selected_points[:, 1]].int(),
                'groups': selected_points_group,
                # 'selected_points': selected_points
            }

        
        return item

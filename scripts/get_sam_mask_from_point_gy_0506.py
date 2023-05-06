import json
import os
import typing
from pathlib import Path

import numpy as np
import torch
import glob
from einops import rearrange
from tools.segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import cv2
import torch
import time
import sys
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')


def init():
    # sam_checkpoint = "samnerf/segment_anything/sam_vit_h_4b8939.pth"
    sam_checkpoint = "tools/segment_anything/sam_vit_h_4b8939.pth"
    
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def flatten(lst):
    """将多维列表变成一维列表的函数"""
    result = []
    for i in lst:
        if isinstance(i, list):
            result.extend(flatten(i))
        else:
            result.append(i)
    return result

def main() -> None:
    # feature = np.load('/data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels/train/sam_features/000000.npy')
    # # 初始化sam
    # predictor = init()
    # predictor.set_feature(feature, [int(3648/4), int(5472/4)])
    # in_points = np.array([[int(3648/8), int(5472/8)]])
    # in_labels = np.array([1])
    # masks, iou_preds, _ = predictor.predict(in_points, in_labels)
    # masks = masks.astype(np.int32)*255
    # cv2.imwrite("mask.png", masks.transpose(1, 2, 0))

    N_total = 20480
    N_each = 1024
    
    H, W = int(3648/4), int(5472/4)
    if sys.argv[1] is not None:
        feature = np.load(sys.argv[1])
    else:
        feature = np.load('/data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels/train/sam_features/000000.npy')

    
    # feature = torch.from_numpy(feature).to(device=device)
    # 初始化sam
    
    in_labels = np.array([1])

    predictor = init()
    predictor.set_feature(feature, [H, W])
    #初始化一个0数组，记录被选择过的mask
    bool_tensor = torch.ones((H, W), dtype=torch.int).to(device=device)
    # ray的计数器
    N_selected = 0
    selected_points = []
    selected_points_group = []

    selected_one = []
    i=0
    visual = torch.zeros((H, W), dtype=torch.int).to(device=device)
    start = time.time()

    while N_selected < N_total:
        random_point = torch.nonzero(bool_tensor)[torch.randint(high=torch.sum(bool_tensor), size=(1,))]
        random_point = random_point.flip(1)
        # random_point = torch.tensor([[400, 400]]).to(device) # W H
        # print(random_point)
        selected_one.append(random_point)
        masks, iou_preds, _ = predictor.predict(random_point.cpu().numpy(), in_labels, return_torch=True)
        
        masks_max = masks[0, torch.argmax(iou_preds),:,:]
        #在mask内选 N_each 个点，若不足 N_each ，则全选
        if masks_max.nonzero().size(0) ==0:
            continue
        elif masks_max.nonzero().size(0) < N_each:
            # print(masks_max.nonzero().size(0))
            select_point = masks_max.nonzero()
            selected_points.append(select_point)
            selected_points_group.append(i * torch.ones(select_point.size(0)).to(device))
            N_selected += masks_max.nonzero().size(0)
        else:
            # print(masks_max.nonzero().size(0))
            select_point = torch.nonzero(masks_max)[torch.randint(high=torch.sum(masks_max), size=(N_each,))]
            selected_points.append(select_point)
            selected_points_group.append(i * torch.ones(select_point.size(0)).to(device))
            N_selected += N_each
        
        masks_max_vis = np.stack([masks_max.cpu().numpy(), masks_max.cpu().numpy(), masks_max.cpu().numpy()], axis=2)
        for point in selected_one:
            #print(point)
            for k in range(point.size(0)):  # W , H
                x, y = int(point[k, 0]), int(point[k, 1])
                for m in range(-2,2):
                    for n in range(-2,2):
                        if x+m < W and x+m >0 and y+n < H and y+n >0:
                            masks_max_vis[(y+n), (x+m)] = np.array([0, 0, 128])
        cv2.imwrite(f"zyq/{sys.argv[2]}_{sys.argv[3]}_mask_final_{i:03d}_{N_selected}.png", masks_max_vis.astype(np.int32)*255)
        visual += masks_max * i
        i += 1

        bool_tensor = bool_tensor * (~masks_max.squeeze(0))
    end = time.time()
    print(end-start)
    
    selected_points = torch.cat(selected_points)
    selected_points_group = torch.cat(selected_points_group)
    print(selected_points.size(0))
    print(selected_points_group.size(0))
    assert selected_points.size(0) == selected_points_group.size(0)
    if selected_points.size(0) > N_total:
        selected_points = selected_points[:N_total]
        selected_points_group = selected_points_group[:N_total]
    print(selected_points.size(0))
    print(selected_points_group.size(0))


    visual = visual.cpu().numpy()
    rgb_array = np.stack([visual, visual, visual], axis=2)*(127/visual.max()+127)
    rgb_array_test = rgb_array
    
    for point in selected_points:
        #for k in range(point.size(0)):  # H, W
        x, y = point[0], point[1]
        if x < H and x >0 and y < W and y >0:
            rgb_array_test[x, y] = np.array([0, 128, 0])

    for point in selected_one:
        # print(point)
        for k in range(point.size(0)):  # W , H
            x, y = int(point[k, 0]), int(point[k, 1])
            for m in range(-5,5):
                for n in range(-5,5):
                    if x+m < W and x+m >0 and y+n < H and y+n >0:
                        rgb_array_test[(y+n), (x+m)] = np.array([0, 0, 128])

    # 显示图像
    cv2.imwrite(f"zyq/{sys.argv[2]}_{sys.argv[3]}_mask_final.png", (rgb_array_test).astype(np.int32))
    print(f"zyq/{sys.argv[2]}_{sys.argv[3]}_mask_final.png")
    # cv2.imwrite("mask_final.png", (visual*255/visual.max()).cpu().numpy().astype(np.int32))
    
    print("Done!")


if __name__ == '__main__':
    main()


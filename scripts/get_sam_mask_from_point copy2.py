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

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


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


    
    H, W = int(3648/4), int(5472/4)
    feature = np.load('/data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels/train/sam_features/000000.npy')
    feature = torch.from_numpy(feature).to(device=device)
    # 初始化sam
    
    predictor = init()
    predictor.set_feature(feature, [H, W])
    in_labels = torch.tensor([[1]]).to(device=device)

    #初始化一个0数组，记录被选择过的mask
    bool_tensor = torch.ones((H, W), dtype=torch.int).to(device=device)
    # ray的计数器
    N_selected = 0
    selected_points = []
    selected_one = []
    i=1
    visual = torch.zeros((H, W), dtype=torch.int).to(device=device)
    while N_selected < 1024:

        random_point = torch.nonzero(bool_tensor)[torch.randint(high=torch.sum(bool_tensor), size=(1,))]
        # random_point = torch.tensor([[400, 400]]).to(device) # W H
        print(random_point)
        selected_one.append(random_point)
        masks, iou_preds, _ = predictor.predict_torch(random_point.unsqueeze(0), in_labels)
        
        masks_max = masks[:,torch.argmax(iou_preds),:,:].squeeze(0)
        #在mask内选128个点，若不足128，则全选
        if masks_max.nonzero().size(0) ==0:
            continue
        elif masks_max.nonzero().size(0) < 128:
            print(masks_max.nonzero().size(0))
            select_point = masks_max.nonzero()
            selected_points.append(select_point)
            N_selected += masks_max.nonzero().size(0)
        else:
            print(masks_max.nonzero().size(0))
            select_point = torch.nonzero(masks_max)[torch.randint(high=torch.sum(masks_max), size=(128,))]
            selected_points.append(select_point)
            N_selected += 128
        cv2.imwrite(f"zyq/mask_{N_selected}.png", masks_max.cpu().numpy().astype(np.int32)*255)
        bool_tensor = bool_tensor * (~masks_max)
        visual += masks_max * i
        i += 1
    visual = visual.cpu().numpy()
    # rgb_array = np.stack([visual, visual, visual], axis=2)*(127/visual.max()+127)
    rgb_array = np.stack([visual, visual, visual], axis=2)*(255)

    rgb_array_test = rgb_array
    for point in selected_one:
        print(point)
        for k in range(point.size(0)):  # H, W
            x, y = point[k, 0], point[k, 1]
            factor=3/4*64/43
            x = int(x * factor)
            y = int(y * factor)

            for m in range(-5,5):
                for n in range(-5,5):
                    if x+m < H and x+m >0 and y+n < W and y+n >0:
                        rgb_array_test[(x+m), (y+n)] = np.array([0, 0, 128])
                        

    for point in selected_points:
        for k in range(point.size(0)):  # H, W
            x, y = point[k, 0], point[k, 1]
            for m in range(-3,3):
                for n in range(-3,3):
                    if x+m < H and x+m >0 and y+n < W and y+n >0:
                        rgb_array_test[x+m, y+n] = np.array([0, 128, 0])

    # 显示图像
    cv2.imwrite("zyq/mask_final.png", (rgb_array_test).astype(np.int32))
    # cv2.imwrite("mask_final.png", (visual*255/visual.max()).cpu().numpy().astype(np.int32))
    
    print("Done!")


if __name__ == '__main__':
    main()


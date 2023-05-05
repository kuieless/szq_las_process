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


    
    H, W = 3648, 5472
    feature = np.load('/data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels/train/sam_features/000000.npy')
    feature = torch.from_numpy(feature).to(device=device)
    # 初始化sam
    predictor = init()
    predictor.set_feature(feature, [int(H/4), int(W/4)])
    
    bool_tensor = torch.ones((H, W))
    

    in_points = torch.tensor([[[int(W/8), int(H/8)]]]).to(device=device)
    in_labels = torch.tensor([[1]]).to(device=device)
    masks, iou_preds, _ = predictor.predict_torch(in_points, in_labels)
    masks_max = masks[:,torch.argmax(iou_preds),:,:].squeeze(0)

    masks_1 = masks[:,0,:,:].squeeze(0)
    masks_2 = masks[:,1,:,:].squeeze(0)
    masks_3 = masks[:,2,:,:].squeeze(0)

    cv2.imwrite("mask1.png", masks_1.cpu().numpy().astype(np.int32)*255)
    cv2.imwrite("mask2.png", masks_2.cpu().numpy().astype(np.int32)*255)
    cv2.imwrite("mask3.png", masks_3.cpu().numpy().astype(np.int32)*255)


    # 设置图像的高和宽
    H = 100
    W = 200

    # 生成布尔值张量
    threshold = 0.5
    bool_tensor = torch.rand((H, W)) < 0.5

    # 随机选取一个点
    random_point = torch.nonzero(bool_tensor)[torch.randint(high=torch.sum(bool_tensor), size=(1,))]

    print(random_point)


    index_tensor = torch.tensor([(h, w) for h in range(H) for w in range(W)])
    random_point = index_tensor[torch.randint(high=index_tensor.shape[0], size=(1,))]
    
    print("Done!")


if __name__ == '__main__':
    main()


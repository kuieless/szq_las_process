import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import time
from tools.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import glob
import os
import tqdm
from pathlib import Path

from argparse import Namespace
import configargparse
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# torch.cuda.set_device(5)
device= 'cuda'


def show_anns1_torch(colors, anns, img_name, hparams):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

    # 创建一个初始的图像张量
    img_shape = (sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3)
    img = torch.ones(img_shape, dtype=torch.float32, device=device)

    mask_center_coor = []
    for ann in sorted_anns:
        if ann['area'] < 0.01*912*1368:
            continue
        m = ann['segmentation']
        # 生成随机颜色
        color_mask = torch.rand(3, device=device) + 0.2
        # color_mask = torch.cat((color_mask, torch.tensor([0.35], device=device)))
        # 使用scatter_函数将颜色应用于图像张量
        img[m] = color_mask
        true_indices = torch.where(torch.from_numpy(m))
        center_x = true_indices[0].float().mean()
        center_y = true_indices[1].float().mean()

        # 中心坐标的值
        center_coordinates = torch.tensor([center_x, center_y]).int()
        mask_center_coor.append(center_coordinates)

    # 将张量转换为图像
    cv2.imwrite(os.path.join(hparams.output_path, 'viz_ori_sam', f"{img_name}.png"), img.cpu().numpy()*255)

    return img.cpu().numpy(), mask_center_coor



def _get_train_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)

    # parser.add_argument('--image_path', type=str, default='/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/9/eval_200000_near/pred_rgb',required=False, help='')
    # parser.add_argument('--sam_feat_path', type=str, default='/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/9/eval_200000_near/sam_features',required=False, help='')
    # parser.add_argument('--output_path', type=str, default='/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/9/eval_200000_near/sam_viz',required=False, help='')
    

    parser.add_argument('--image_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/rgbs',required=False, help='')
    parser.add_argument('--sam_feat_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/sam_features',required=False, help='')
    parser.add_argument('--output_path', type=str, default='zyq/get_instance_mask',required=False, help='')
    

    return parser.parse_args()


def hello(hparams: Namespace) -> None:
    file_name='DJi'

    sam_checkpoint = "tools/segment_anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32)

    predictor = SamPredictor(sam)

    imgs = []
    for ext in ('*.png'):
        imgs.extend(glob.glob(os.path.join(hparams.image_path, ext)))
    imgs.sort()
    imgs = imgs[1:]


    features = []
    for ext in ('*.npy'):
        features.extend(glob.glob(os.path.join(hparams.sam_feat_path, ext)))
    features.sort()
    features = features[1:]



    colors = []
    for i in range(len(imgs)):
        colors.append(np.random.random((1,3)).tolist()[0])
    colors = np.clip(colors, 0, 0.95)

    Path(hparams.output_path).mkdir(exist_ok=True)
    Path(os.path.join(hparams.output_path,'image_cat')).mkdir(exist_ok=True)
    Path(os.path.join(hparams.output_path,'viz_ori_sam')).mkdir(exist_ok=True)
    Path(os.path.join(hparams.output_path,'instance_mask_vis')).mkdir(exist_ok=True)
    Path(os.path.join(hparams.output_path,'instance_mask_vis_cat')).mkdir(exist_ok=True)




    in_labels = np.array([1])

    
    for i in tqdm.tqdm(range(len(imgs))):
        img_name = imgs[i].split('/')[-1][:6]
        # if int(img_name) < 203 or int(img_name) > 239:
        #     continue
        image = cv2.imread(imgs[i])
        # image_size = 
        h, w, _ = image.shape
        if w > 3000:
            image = cv2.resize(image, (int(w/4), int(h/4)), cv2.INTER_AREA)
            h, w, _ = image.shape
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(features[i])
        feature = torch.from_numpy(np.load(features[i])) #.to(device)
        
        masks_auto = mask_generator.generate(image1, feature[0])

        mask_vis, mask_center_coor = show_anns1_torch(colors, masks_auto, img_name, hparams)
        image_cat = image*0.6+ (mask_vis*255)*0.4
        cv2.imwrite(os.path.join(hparams.output_path,'image_cat', f"{img_name}.png"), image_cat)
        
        predictor.set_feature(feature[0], [h, w])

        instance_mask = torch.zeros((h,w),dtype=torch.int32).to(device)
        for j, mask_center in enumerate(mask_center_coor):
            
            mask_center = mask_center.unsqueeze(0).flip(1)

            masks, iou_preds, _ = predictor.predict(mask_center.cpu().numpy(), in_labels, return_torch=True)
            masks_max = masks[:, torch.argmax(iou_preds),:,:].squeeze(0)
            
            overlap = masks_max & instance_mask.bool()
            overlap_area = overlap.sum().item()
            if (instance_mask[masks_max].nonzero()).sum() == 0:
                instance_mask[masks_max] = int(j)+1
                continue
            unique_c, counts_c = torch.unique(instance_mask[masks_max].nonzero(), return_counts=True)
            
            labels_max_idx = torch.argmax(counts_c)

            if overlap_area > 0.5:
                instance_mask[masks_max] = unique_c[labels_max_idx]
            else:
                instance_mask[masks_max] = int(j)+1


            # 计算重叠区域的面积
            overlap_area = overlap.sum().item()
            # new_mask = torch.zeros((h,w),dtype=torch.int32)
            # new_mask[masks_max] = int(j)+1
            # total_size = new_mask[masks_max].shape[0]
            # unique_c, counts_c = torch.unique(instance_mask[masks_max].nonzero(), return_counts=True)
            # for m, unique_m in enumerate(unique_c):
            #     if (instance_mask == unique_m).sum() == counts_c[m]:  # 判断在new mask内的点是不是所有点
            #         new_mask[masks_max]= unique_m  # 如果是，新的mask变成 旧的id
            #     elif total_size == counts_c[m]:
            #         new_mask[masks_max]= unique_m
            #     else:
            #         if counts_c[m] / total_size >0.3: # 该id占mask比例
            #             new_mask[masks_max] = unique_m
            #         else:
            #             new_mask[masks_max] = int(j)+1
            # instance_mask[masks_max]=new_mask[masks_max]



        
        instance_mask_vis = torch.zeros((h,w,3))
        for k in enumerate(torch.unique(instance_mask)):
            color_mask = torch.rand(3, device=device) + 0.2
            instance_mask_vis[instance_mask==k] = color_mask

        cv2.imwrite(os.path.join(hparams.output_path, 'instance_mask_vis', f"{img_name}.png"), instance_mask_vis.cpu().numpy()*255)


        instance_mask_vis = image*0.6+ (instance_mask_vis.cpu().numpy()*255)*0.4
        cv2.imwrite(os.path.join(hparams.output_path,'instance_mask_vis_cat', f"{img_name}.png"), image_cat)
        




            
            
                    
    print('done')


if __name__ == '__main__':
    hello(_get_train_opts())
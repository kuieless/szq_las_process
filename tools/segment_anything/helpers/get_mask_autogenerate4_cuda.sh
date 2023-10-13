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

torch.cuda.set_device(6)
device= 'cpu'


# #sam在demo中的版本
def show_anns1(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))

    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_anns1_torch(colors, anns, img_name, hparams):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

    # 创建一个初始的图像张量
    img_shape = (sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3)
    img = torch.ones(img_shape, dtype=torch.float32, device=device)

    # 设置透明度通道
    # img[:, :, 3] = 0

    for ann in sorted_anns:
        m = ann['segmentation']

        # 生成随机颜色
        color_mask = torch.rand(3, device=device) + 0.2
        # color_mask = torch.cat((color_mask, torch.tensor([0.35], device=device)))

        # 使用scatter_函数将颜色应用于图像张量
        img[m] = color_mask

    # 将张量转换为图像
    # img_pil.save(os.path.join(hparams.output_path, 'viz_ori_sam', f"{img_name}.png"))
    cv2.imwrite(os.path.join(hparams.output_path, 'viz_ori_sam', f"{img_name}.png"), img.numpy()*255)

    return img.numpy()


def show_anns2(colors, anns, img_name, hparams):
    
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 3))  # 255 denotes white
    
    mask = np.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 1))
    mask_visual = np.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 3))
    visual = np.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 3))
    
    temp_mask_list=[]
    save =0
    for i in range(len(sorted_anns)):
    # for i in range(12):

        m = sorted_anns[i]['segmentation']
        visual[m] = np.random.random((1,3))
        
        overlap = (m * mask[:,:,0]).astype(bool)
        if overlap.sum() / m.sum() < 0.2:
            if save==90:
                mask_visual = np.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 3))
                
            if save>=102:
                break

            save += 1
            mask[m] = i + 1
            temp_mask = torch.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 1))
            temp_mask[m] = i + 1
            temp_mask_list.append(temp_mask)
            mask_visual[m] = np.random.random((1,3))
            # if save>=90:
                # cv2.imwrite(f"./tools/segment_anything/SAM_mask_autogenerate_{file_name}/"+"mask_%04d.png" % i, mask_visual*255)

        # cv2.imwrite(f"./tools/segment_anything/SAM_mask_autogenerate_{file_name}/"+"visual_%04d.png" % i, visual*255)

    Path(os.path.join(hparams.output_path,'viz_no_filter')).mkdir(exist_ok=True)
    Path(os.path.join(hparams.output_path,'viz_overlap_filter')).mkdir(exist_ok=True)

    cv2.imwrite(os.path.join(hparams.output_path, 'viz_no_filter', f"{img_name}.png"), visual*255)
    cv2.imwrite(os.path.join(hparams.output_path, 'viz_overlap_filter', f"{img_name}.png"), mask_visual*255)
    

    
    # temp_mask_list=torch.cat(temp_mask_list, dim=-1)
    # np.save(os.path.join(hparams.output_path, 'sam_features_filter', f"{img_name}.npy"), temp_mask_list.numpy())

    # labels = np.load('/data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/000027_sam.npy')


    
    # for i in range(len(sorted_anns)):
    #     m = anns[i]['segmentation']
    #     mask = mask + m[:,:,np.newaxis] * np.random.random((1,3))
    return visual








def _get_train_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)

    # parser.add_argument('--image_path', type=str, default='/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/9/eval_200000_near/pred_rgb',required=False, help='')
    # parser.add_argument('--sam_feat_path', type=str, default='/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/9/eval_200000_near/sam_features',required=False, help='')
    # parser.add_argument('--output_path', type=str, default='/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/9/eval_200000_near/sam_viz',required=False, help='')
    

    parser.add_argument('--image_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/rgbs',required=False, help='')
    parser.add_argument('--sam_feat_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/sam_features',required=False, help='')
    parser.add_argument('--output_path', type=str, default='/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/10/eval_200000_ori/sam_viz',required=False, help='')
    

    return parser.parse_args()


def hello(hparams: Namespace) -> None:
    file_name='DJi'

    sam_checkpoint = "tools/segment_anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32)

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


    
    for i in tqdm.tqdm(range(len(imgs))):
        img_name = imgs[i].split('/')[-1][:6]
        if int(img_name) < 203 or int(img_name) > 239:
            continue
        image = cv2.imread(imgs[i])
        # image_size = 
        w, h, _ = image.shape
        if w > 3000:
            image = cv2.resize(image, (int(h/4), int(w/4)), cv2.INTER_AREA)
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(features[i])
        feature = torch.from_numpy(np.load(features[i]))
        masks = mask_generator.generate(image1, feature[0])
        
        # mask = show_anns1(masks)
        mask = show_anns1_torch(colors, masks, img_name, hparams)

        # mask = show_anns2(colors, masks, img_name, hparams)
        # image_cat = np.concatenate([image, mask*255], axis=1)
        image_cat = image*0.6+ (mask*255)*0.4
        cv2.imwrite(os.path.join(hparams.output_path,'image_cat', f"{img_name}.png"), image_cat)
                    
    print('done')


if __name__ == '__main__':
    hello(_get_train_opts())
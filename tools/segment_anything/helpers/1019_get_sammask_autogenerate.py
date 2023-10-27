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
from PIL import Image

# torch.cuda.set_device(7)
device= 'cuda'


def save_mask_anns_torch(colors, anns, img_name, hparams, id):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=False)

    # 创建一个初始的图像张量
    img_shape = (sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1])
    img = torch.zeros(img_shape, dtype=torch.int16, device=device)

    # 设置透明度通道
    # img[:, :, 3] = 0

    # id = 1
    for ann in sorted_anns:
        if ann['area'] < 0.01*img_shape[0]*img_shape[1]:
            continue
        m = ann['segmentation']

        img[m] = id
        id = id + 1

    return img, id

def visualize_labels(labels_tensor):
    non_zero_labels = labels_tensor[labels_tensor != 0]
    unique_labels = torch.unique(non_zero_labels)
    num_labels = len(unique_labels)

    # 创建颜色映射
    label_colors = torch.rand((num_labels, 3))  # 生成随机颜色
    colored_image = torch.zeros((labels_tensor.size(0), labels_tensor.size(1), 3))  # 创建空的彩色图像

    for i, label in enumerate(unique_labels):
        label_mask = (labels_tensor == label)
        colored_image[label_mask] = label_colors[i]

    # 将 PyTorch 张量转换为 NumPy 数组
    colored_image_np = (colored_image.numpy() * 255).astype(np.uint8)

    return colored_image_np

def _get_train_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)

    # parser.add_argument('--image_path', type=str, default='/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/9/eval_200000_near/pred_rgb',required=False, help='')
    # parser.add_argument('--sam_feat_path', type=str, default='/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/9/eval_200000_near/sam_features',required=False, help='')
    # parser.add_argument('--output_path', type=str, default='/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/9/eval_200000_near/sam_viz',required=False, help='')
    

    parser.add_argument('--image_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/rgbs',required=False, help='')
    parser.add_argument('--sam_feat_path', type=str, default='/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/sam_features',required=False, help='')
    parser.add_argument('--output_path', type=str, default='zyq/1027_get_instance_mask_train',required=False, help='')
    

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
    Path(os.path.join(hparams.output_path,'instances_mask')).mkdir(exist_ok=True)
    Path(os.path.join(hparams.output_path,'instances_mask_vis')).mkdir(exist_ok=True)
    Path(os.path.join(hparams.output_path,'image_cat')).mkdir(exist_ok=True)

    used_files = []
    for ext in ('*.png', '*.jpg'):
        used_files.extend(glob.glob(os.path.join('/data/yuqi/Datasets/DJI/Yingrenshi_20230926_subset/train/rgbs', ext)))
    used_files.sort()
    process_item = [Path(far_p).stem for far_p in used_files]

    id = 1 
    for i in tqdm.tqdm(range(len(imgs))):
        img_name = imgs[i].split('/')[-1][:6]
        if img_name not in process_item and 'val' not in hparams.image_path:
            continue
        # if int(img_name) < 203 or int(img_name) > 239:
        #     continue
        image = cv2.imread(imgs[i])
        # image_size = 
        w, h, _ = image.shape
        if w > 3000:
            image = cv2.resize(image, (int(h/4), int(w/4)), cv2.INTER_AREA)
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(features[i])
        feature = torch.from_numpy(np.load(features[i]))

        ### NOTE: 有时候会报维度不匹配的错误，修改下面的代码
        masks = mask_generator.generate(image1, feature[0])
        # masks = mask_generator.generate(image1, feature)
        
        mask, id = save_mask_anns_torch(colors, masks, img_name, hparams, id)
        mask_vis = visualize_labels(mask)
        
        Image.fromarray(mask.cpu().numpy().astype(np.uint8)).save(os.path.join(hparams.output_path, 'instances_mask', f"{img_name}.png"))
        

        cv2.imwrite(os.path.join(hparams.output_path, 'instances_mask_vis', f"{img_name}.png"), mask_vis)

        image_cat = image*0.6+ mask_vis*0.4
        cv2.imwrite(os.path.join(hparams.output_path,'image_cat', f"{img_name}.png"), image_cat)
                    
    print('done')


if __name__ == '__main__':
    hello(_get_train_opts())
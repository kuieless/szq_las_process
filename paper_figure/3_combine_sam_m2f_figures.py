import numpy as np
from pycocotools import mask
from PIL import Image
import torch
import matplotlib.pyplot as plt
import glob
import os
import time
import sys
from pathlib import Path
from tqdm import tqdm
import cv2
from tools.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from argparse import Namespace
import configargparse

# torch.cuda.set_device(6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_id(sam, m2f):
    # print('mask', i)
    # sam = torch.from_numpy(sam).long().to(device)
    sam = sam.to(torch.long)
    labeled_sam = torch.where(sam==1, m2f, sam)
    
    unique, counts = torch.unique(labeled_sam, return_counts=True)
    # print('labeled', unique, counts)

    # label the whole mask with the most frequent label
    if len(unique) > 1:
        unique = unique[1:]
        counts = counts[1:]

        # add by zyq:
        # attach the labels only when the max_label_class > 10% * mask_area
        sam_mask_unique, sam_mask_counts = torch.unique(sam, return_counts=True)
        counts_total = sam_mask_counts[sam_mask_unique==1]
        couts_max_label = counts.max()
        # if couts_max_label / counts_total > 0.1:
        if couts_max_label / counts_total > 0.2:    #20231018 改成>0.7, 避免
            id_max_label = unique[counts.argmax()]
        else:
            id_max_label = 0
        
        # id_max_label = unique[counts.argmax()]
        
        result = torch.where(sam==1, id_max_label, sam)
    else:
        result = torch.where(sam==1, unique[counts.argmax()], sam)

    return result
def get_mask(dict, semantics):
    if len(dict) == 0:
        return

    mask = torch.zeros_like((dict[0]['segmentation'])).to(torch.long)
    for i in range(len(dict)):
        id = dict[i]['id']
        mask = torch.where(mask==0, id, mask) # avoid overwriting
    mask = torch.where(mask==0, semantics, mask)

    # mask = np.zeros((dict[0]['segmentation'].shape[0], dict[0]['segmentation'].shape[1]))
    # for i in range(len(dict)):
    #     id = dict[i]['id']
    #     mask = np.where(mask==0, id, mask) # avoid overwriting
    # mask = np.where(mask==0, semantics, mask)
    return mask.cpu().numpy()
def custom2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]             # cluster       black
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 0, 0]           # building      red
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [192, 192, 192]       # road          grey  
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [192, 0, 192]         # car           light violet
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 128, 0]           # tree          green
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [128, 128, 0]         # vegetation    dark green
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 255, 0]         # human         yellow
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [135, 206, 250]       # sky           light blue
    mask_rgb[np.all(mask_convert == 8, axis=0)] = [0, 0, 128]           # water         blue
    mask_rgb[np.all(mask_convert == 9, axis=0)] = [252,230,201]         # ground        egg
    mask_rgb[np.all(mask_convert == 10, axis=0)] = [128, 64, 128]       # mountain      dark violet

    # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb

def _get_train_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)

    # parser.add_argument('--sam_features_path', type=str, default='',required=False, help='')
    parser.add_argument('--labels_m2f_path', type=str, default='',required=False, help='')
    parser.add_argument('--rgbs_path', type=str, default='',required=False, help='')
    parser.add_argument('--output_path', type=str, default='/data/yuqi/code/GP-NeRF-semantic/paper_figure/3_semantic_fusion_1117',required=False, help='')
    
    return parser.parse_args()



def hello(hparams: Namespace) -> None:
    
    save_path_o = hparams.output_path
    save_path = os.path.join(save_path_o, 'labels_merge')
    save_path_vis = os.path.join(save_path_o, 'labels_merge_vis')
    save_path_alpha = os.path.join(save_path_o, 'alpha')
    Path(save_path).mkdir(exist_ok=True,parents=True)
    Path(save_path_vis).mkdir(exist_ok=True)
    Path(save_path_alpha).mkdir(exist_ok=True)

    sam_checkpoint = "tools/segment_anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=128)
    


    img_name = Path('/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/rgbs/000151.jpg').stem
    img_p = '/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/rgbs/000151.jpg'
    image = cv2.imread(img_p)
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = cv2.resize(image1,(1368,912))

    m2f = Image.open('/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1018_far0.3/project_to_ori_gt_only_sam_filter_sam_replace_building/replace_building_label/000151.png')


    semantics = torch.from_numpy(np.array(m2f)).long().to(device)

    # load_dict = mask_generator.generate(image1, feature[0])
    # load_dict = mask_generator.generate(image1, feature)
    load_dict = mask_generator.generate(image1)


    dict = [{'segmentation': torch.tensor(load_dict[k]['segmentation']).to(device), 'id': get_id(torch.tensor(load_dict[k]['segmentation']).to(device), semantics)} for k in range(len(load_dict))]

    _mask = get_mask(dict, semantics)
    Image.fromarray(_mask.astype(np.uint16)).save(os.path.join(save_path, img_name+".png"))

    mask_rgb = custom2rgb(_mask)
    Image.fromarray(mask_rgb).save(os.path.join(save_path_vis, img_name+".png"))
    
    
    Image.fromarray((0.5*mask_rgb+0.5*image1).astype(np.uint8)).save(os.path.join(save_path_alpha, img_name+".png"))


    


if __name__ == '__main__':
    hello(_get_train_opts())
#
#  这个代码是用于处理大疆智图空三导出的xml文件
#  首先，将xml的gos坐标系转换成cecf坐标系
#  再进行缩放旋转，使其朝向+x轴，
#  ps：根据xml的yaw pitch roll赋予label，在训练中去除黑影（脚架）
#


from argparse import Namespace
from pathlib import Path
import numpy as np
import configargparse
import os
from tqdm import tqdm
import shutil
import torch
import trimesh
import random
import math
import cv2
import sys
from glob import glob
sys.path.append(".")
print("sys:", sys.path)
print("cwd:", os.getcwd())
# from dji.visual_poses import visualize_poses, load_poses
from bs4 import BeautifulSoup

import json
from PIL import Image
from PIL.ExifTags import TAGS
import exifread

def _get_opts():

    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--dataset_path', default='/data/yuqi/longhua_block2_test',type=str, required=False)  # 暂时没用
    parser.add_argument('--output_path', default='/data/yuqi/longhua_block2_test/output',type=str, required=False)
    return parser.parse_known_args()[0]




def main(hparams):
    # initialize
    dataset_pth= hparams.dataset_path
    output_path = Path(hparams.output_path)
    if not os.path.exists(hparams.output_path):
        output_path.mkdir(parents=True)
    (output_path / 'left').mkdir(parents=True, exist_ok=True)
    (output_path / 'right').mkdir(parents=True, exist_ok=True)
    (output_path / 'others').mkdir(parents=True, exist_ok=True)
    
    used_files = []
    for ext in ('*.pt', '*.jpg'):
        used_files.extend(glob(os.path.join(dataset_pth, 'train', 'metadata', ext)))
    used_files.sort()

    for metadata_path in tqdm(used_files):
        file_name = Path(metadata_path).stem

        img_path = os.path.join(dataset_pth, 'train', 'rgbs', file_name+'.jpg')


        img1 = cv2.imread(img_path)
        metadata = torch.load(metadata_path, map_location='cpu')
        left_or_right = metadata['left_or_right']
        if left_or_right == 'left':
            cv2.imwrite(os.path.join(output_path, 'left', file_name+'.jpg'), img1)
        elif left_or_right == 'right':
            cv2.imwrite(os.path.join(output_path, 'right', file_name+'.jpg'), img1)
        else:
            cv2.imwrite(os.path.join(output_path, 'others', file_name+'.jpg'), img1)

    print('Finish')


if __name__ == '__main__':
    main(_get_opts())
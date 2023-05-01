import argparse
import glob
import multiprocessing as mp
import os
import gzip
import sys
import tempfile
import time
import warnings
import torch
import cv2
import numpy as np
import tqdm
from pathlib import Path
import math

def main():
    dataset_path = '/data/yuqi/Datasets/MegaNeRF/UrbanScene3D/campus/campus-pixsfm'
    # dataset_path = '/data/yuqi/Datasets/MegaNeRF/Mill19/building/building-pixsfm'


    used_files = []
    for ext in ('*.JPG', '*.jpg'):
        # used_files.extend(glob.glob(os.path.join(dataset_path, 'train', 'rgbs', ext)))
        used_files.extend(glob.glob(os.path.join(dataset_path, 'val', 'rgbs', ext)))
    used_files.sort()
    (Path(used_files[0]).parent.parent / f"resize_rgbs").mkdir(exist_ok=True)
    for path in tqdm.tqdm(used_files):
    
        # #Image
        # rgbs = Image.open(path).convert('RGB')
        # size = rgbs.size
        # assert size[0] % 4 == 0
        # assert size[1] % 4 == 0
        # # print(f'H: {size[0]}, W: {size[1]}')
        # # print(f'H resize: {size[0] /4}, W resize: {size[1]/4}')
        # rgbs = rgbs.resize((int(size[0] / 4), int(size[1]/4)), Image.LINEAR)
        # save_path = Path(path)
        # save_path_rgbs = save_path.parent.parent / 'resize_rgbs' / save_path.name
        # Image.fromarray(np.array(rgbs)).save(save_path_rgbs)

        #cv2 
        rgbs = cv2.imread(path)
        size = rgbs.shape
        assert size[0] % 4 == 0
        assert size[1] % 4 == 0
        rgbs = cv2.resize(rgbs, (int(size[1]/4), int(size[0]/4)), interpolation=cv2.INTER_LINEAR)
        save_path = Path(path)
        save_path_rgbs = str(save_path.parent.parent / 'resize_rgbs' / save_path.name)
        cv2.imwrite(save_path_rgbs, rgbs)


if __name__ == '__main__':
    main()

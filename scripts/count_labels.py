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
from typing import Optional
from zipfile import ZipFile

import torch.nn.functional as F
from PIL import Image
import sys
from tools.unetformer.uavid2rgb import remapping

def main():
    dataset_path = f'/data/yuqi/Datasets/MegaNeRF/{sys.argv[1]}/{sys.argv[2]}/{sys.argv[2]}-labels'

    CLASSES = ('Cluster', 'Building', 'Road', 'Car', 'Tree', 'Vegetation', 'Human', 'Sky', 'Water', 'Ground', 'Mountain')
    counts = np.zeros((len(CLASSES)))
    used_files = []
    used_files.extend(glob.glob(os.path.join(dataset_path, 'val', 'labels_gt', '*.png')))
    used_files.sort()
    print(used_files)
    if 'residence'in dataset_path:
        used_files=used_files[:19]
    elif 'building'in dataset_path or 'campus'in dataset_path:
        used_files=used_files[:10]
    Path('zyq').mkdir(exist_ok=True,parents=True)
    
    with (Path('zyq') / f'{sys.argv[2]}_labels_count.txt').open('w') as f:
        for path in tqdm.tqdm(used_files):
            labels = Image.open(path)    #.convert('RGB')
            size = labels.size
            labels = np.asarray(labels).copy()
            labels = remapping(labels)
            for i in range(len(CLASSES)):
                count = len(labels[labels==i])
                # print(count)
                counts[i] = counts[i] + count
        total_num = size[0]*size[1]*len(used_files)
        total_count = 0
        for i in range(len(CLASSES)):
            total_count += counts[i]
        assert  total_num == total_count

        for i in range(len(CLASSES)):
            class_id = CLASSES[i]
            f.write(f'{class_id:<12}: {counts[i]:<10}   {(counts[i] / total_count * 100)} % \n')
    print(len(used_files))
    
    print(Path('zyq') / f'{sys.argv[2]}_labels_count.txt')
    print('done')



if __name__ == '__main__':
    main()


from plyfile import PlyData
import numpy as np
from tqdm import tqdm

import laspy
import pyproj
import torch
import datetime
from datetime import timedelta
from datetime import datetime
from pathlib import Path
from argparse import Namespace
import configargparse

import pickle
import os

def _get_opts():

    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--dataset_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan',type=str, required=False)
    parser.add_argument('--las_path', default='/data/yuqi/Datasets/DJI/origin/DJI_20230726_xiayuan_data/H2.las',type=str, required=False)
    parser.add_argument('--las_output_path', default='./test',type=str, required=False)
    parser.add_argument('--num_val', type=int, default=20, help='Number of images to hold out in validation set')
    parser.add_argument('--start', type=int, help='')
    parser.add_argument('--end', type=int, help='')

    return parser.parse_known_args()[0]

def main(hparams):
    dataset_path = Path(hparams.dataset_path)
    train_path_candidates = sorted(list((dataset_path / 'train' / 'image_metadata').iterdir()))
    train_paths = [train_path_candidates[i] for i in range(0, len(train_path_candidates))]
    val_path_candidates = sorted(list((dataset_path / 'val' / 'image_metadata').iterdir()))
    val_paths = [val_path_candidates[i] for i in range(0, len(val_path_candidates))]
    total_num=len(train_paths+val_paths)
    print("process .las file, wait a minute...")
    # Load the LAS file
    in_file = laspy.file.File(hparams.las_path, mode='r')

    x, y, z = in_file.x, in_file.y, in_file.z
    r = (in_file.red / 65536.0 * 255).astype(np.uint8)
    g = (in_file.green / 65536.0 * 255).astype(np.uint8)
    b = (in_file.blue / 65536.0 * 255).astype(np.uint8)
    lidars = np.array(list(zip(x, y, z, r, g, b)))
    
    las_output_path = Path(hparams.las_output_path)
    if not os.path.exists(hparams.las_output_path):
        las_output_path.mkdir(parents=True)
    # np.save(hparams.las_output_path +'lidars', lidars)
    # intensity=in_file.intensity
    
    diff_list = []
    
    # for i in tqdm(range(total_num)):
    for i in tqdm(range(hparams.start, hparams.end), desc="calculate GPS time"):
        if i % int(total_num / hparams.num_val) == 0:
            split_dir = dataset_path / 'val'
        else:
            split_dir = dataset_path / 'train'

        image_metadata = torch.load(str(split_dir / 'image_metadata' / '{0:06d}.pt'.format(i)))
        time = image_metadata['meta_tags']['EXIF DateTimeOriginal'].values
        date_str, time_str = time.split()
        year, month, day = date_str.split(":")
        hour, minute, second = time_str.split(":")
        current_seconds = int(hour) * 3600 + int(minute) * 60 + int(second)
        if i == hparams.start:
            origin_second = current_seconds
            diff=0
            diff_list.append(0)
        else:
            previous_diff = diff
            diff = current_seconds-origin_second
            # diff_list.append(diff-previous_diff)
            diff_list.append(diff)
    
    diff_gps_time = np.array(in_file.gps_time - in_file.gps_time[0])
    points_lidar_list = []
    
    for j in tqdm(range(1, len(diff_list)), desc="save lidars into file"):
        bool_mask = (diff_gps_time >= int(diff_list[j-1])) * (diff_gps_time < int(diff_list[j]))
        points_lidar_list.append(lidars[bool_mask])
    
    bool_mask = diff_gps_time >= int(diff_list[-1])
    points_lidar_list.append(lidars[bool_mask])
    
    with open(str(las_output_path / 'points_lidar_list.pkl'), "wb") as file:
        pickle.dump(points_lidar_list, file)
    print(f"lidars save at {str(las_output_path / 'points_lidar_list.pkl')}")

    print('done')

if __name__ == '__main__':
    main(_get_opts())
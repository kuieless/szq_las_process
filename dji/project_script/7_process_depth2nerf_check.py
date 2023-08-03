
from plyfile import PlyData
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2
import configargparse
from pathlib import Path

import json
from PIL import Image
from PIL.ExifTags import TAGS
import exifread
import math
import torch
from dji.process_dji_v8_color import euler2rotation, rad
from torchvision.utils import make_grid
import os
from dji.visual_poses import visualize_poses, load_poses


def visualize_scalars(scalar_tensor: torch.Tensor) -> np.ndarray:
        to_use = scalar_tensor.view(-1)
        while to_use.shape[0] > 2 ** 24:
            to_use = to_use[::2]

        mi = torch.quantile(to_use, 0.05)
        ma = torch.quantile(to_use, 0.95)

        scalar_tensor = (scalar_tensor - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
        scalar_tensor = scalar_tensor.clamp_(0, 1)

        scalar_tensor = ((1 - scalar_tensor) * 255).byte().numpy()  # inverse heatmap
        return cv2.cvtColor(cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

def _get_opts():

    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--original_images_path', default='/data/yuqi/Datasets/DJI/origin/DJI_20230726_xiayuan_data/original_image',type=str, required=False)
    parser.add_argument('--original_images_list_json_path', default='/data/yuqi/Datasets/DJI/origin/DJI_20230726_xiayuan_data/original_image_list.json',type=str, required=False)
    parser.add_argument('--infos_path', default='/data/yuqi/Datasets/DJI/origin/DJI_20230726_xiayuan_data/BlocksExchangeUndistortAT.xml',type=str, required=False)
    parser.add_argument('--output_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan',type=str, required=False)
    parser.add_argument('--resume', default=True, action='store_false')  # debug
    # parser.add_argument('--resume', default=False, action='store_true')  # run
    parser.add_argument('--num_val', type=int, default=20, help='Number of images to hold out in validation set')
    parser.add_argument('--down_scale', type=int, default=4, help='')
    return parser.parse_known_args()[0]

def main(hparams):

    down_scale = hparams.down_scale

    output_path = Path(hparams.output_path)
    original_images_path = Path(hparams.original_images_path)

    # load the pose and image name from .xml file
    root = ET.parse(hparams.infos_path).getroot()
    xml_pose = np.array([[ float(pose.find('Center/x').text),
                        float(pose.find('Center/y').text),
                        float(pose.find('Center/z').text),
                        float(pose.find('Rotation/Omega').text),
                        float(pose.find('Rotation/Phi').text),
                        float(pose.find('Rotation/Kappa').text)] for pose in root.findall('Block/Photogroups/Photogroup/Photo/Pose')])
    images_name = [Images_path.text.split("\\")[-1] for Images_path in root.findall('Block/Photogroups/Photogroup/Photo/ImagePath')]

    # load "original_images_list.json"
    with open(hparams.original_images_list_json_path, "r") as file:
        json_data = json.load(file)

    sorted_json_data = sorted(json_data, key=lambda x: x["origin_path"])  # 虽然json_data中的顺序也是按飞行顺序来，还是做个sorted保险一些
    sorted_process_data = []
    for i in tqdm(range(len(sorted_json_data)), desc="load meta_data"):
        sorted_process_line = {}
        json_line = json_data[i]
        id = json_line['id']
        index = images_name.index(id+'.jpg')
        sorted_process_line['images_name'] = images_name[index]
        
        path_segments = json_line['origin_path'].split('/')
        last_two_path = '/'.join(path_segments[-2:])
        sorted_process_line['original_image_name'] = last_two_path
        sorted_process_line['pose'] = xml_pose[index, :]
        sorted_process_data.append(sorted_process_line)

    xml_pose_sorted = np.array([x["pose"] for x in sorted_process_data])
    images_name_sorted = [x["images_name"] for x in sorted_process_data]

    # process the poses info to the requirement of nerf
    pose_dji = xml_pose_sorted
    camera_positions = pose_dji[:,0:3]#.astype(np.float32)
    camera_rotations = pose_dji[:, 3:6]#.astype(np.float32)


    # # NOTE 下面代码用来可视化pose
    c2w_R = []
    for i in range(len(camera_rotations)):
        R_temp = euler2rotation(camera_rotations[i])
        c2w_R.append(R_temp)

    ZYQ = torch.DoubleTensor([[0, 0, -1],
                             [0, 1, 0],
                             [1, 0, 0]])
    ZYQ_1 = torch.DoubleTensor([[1, 0, 0],
                              [0, math.cos(rad(135)), math.sin(rad(135))],
                              [0, -math.sin(rad(135)), math.cos(rad(135))]])

    c2w = []
    for i in range(len(c2w_R)):
        temp = np.concatenate((c2w_R[i], camera_positions[i:i + 1].T), axis=1)
        temp = np.concatenate((temp[:,0:1], -temp[:,1:2], -temp[:,2:3], temp[:,3:]), axis=1)
        temp = torch.hstack((ZYQ @ temp[:3, :3], ZYQ @ temp[:3, 3:]))
        temp = torch.hstack((ZYQ_1 @ temp[:3, :3], ZYQ_1 @ temp[:3, 3:]))
        temp = temp.numpy()
        c2w.append(temp)
    c2w = np.array(c2w)

    min_position = np.min(c2w[:,:, 3], axis=0)
    max_position = np.max(c2w[:,:, 3], axis=0)
    print('Coord range: {} {}'.format(min_position, max_position))
    origin = (max_position + min_position) * 0.5
    dist = (torch.tensor(c2w[:,:, 3]) - torch.tensor(origin)).norm(dim=-1)
    diagonal = dist.max()
    scale = diagonal.numpy()
    c2w[:,:, 3] = (c2w[:,:, 3] - origin) / scale
    assert np.logical_and(c2w[:,:, 3] >= -1, c2w[:,:, 3] <= 1).all()

    
    visualize_poses(c2w[290:310])


    NOTE: 下面代码是可视化所有深度图
    for i, rgb_name in enumerate(tqdm(images_name_sorted)):
        # # if i %50 !=0:
        # # if i <= 1750:
        # if i <= 1500 or i > 1750 :
        #     continue
        if i % int(camera_positions.shape[0] / hparams.num_val) == 0:
            split_dir = output_path / 'val'
        else:
            split_dir = output_path / 'train'

        depth = np.load(str(split_dir) + '/depth_dji/{0:06d}.npy'.format(i))
        depth = torch.HalfTensor(depth).float()
        invalid_mask = torch.isinf(depth_dji)
        depth_dji = torch.from_numpy(visualize_scalars(depth, invalid_mask))
        
        img_list=[]
        img_list.append(depth_dji)
        img_list = torch.stack(img_list).permute(0,3,1,2)
        img = make_grid(img_list, nrow=1)
        img_grid = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        Image.fromarray(img_grid).save('/data/yuqi/code/GP-NeRF-semantic/dji/project_script/check/'+'{0:06d}.jpg'.format(i))
        

if __name__ == '__main__':
    main(_get_opts())
import datetime
import os
import random
import traceback
from argparse import Namespace
from pathlib import Path
import math
import numpy as np
import torch
from torch.distributed.elastic.multiprocessing.errors import record
import configargparse
import trimesh
import json
from scripts.colmap_to_mega_nerf import * #read_model, qvec2rotmat, RDF_TO_DRB


def visualize_poses(poses1, poses2=None, size=0.01, flipz=True):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]
    
    flag = -1 if flipz else 1

    for pose in poses1:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]

        a = pos - size * pose[:3, 0] - size * pose[:3, 1] + flag * size * pose[:3, 2]
        b = pos + size * pose[:3, 0] - size * pose[:3, 1] + flag * size * pose[:3, 2]
        c = pos + size * pose[:3, 0] + size * pose[:3, 1] + flag * size * pose[:3, 2]
        d = pos - size * pose[:3, 0] + size * pose[:3, 1] + flag * size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)

        segs1 = np.array([[pos, a], [pos, b], [a, b]])
        segs1 = trimesh.load_path(segs1)
        segs1.colors = np.array([[128, 0, 0]] * len(segs1.entities))
        objects.append(segs1)
        segs3 = np.array([[pos, c],[pos, d], [c, d]])
        segs3 = trimesh.load_path(segs3)
        segs3.colors = np.array([[0, 128, 0]] * len(segs3.entities))
        objects.append(segs3)
        segs4 = np.array([[b, c]])
        segs4 = trimesh.load_path(segs4)
        segs4.colors = np.array([[0, 0, 128]] * len(segs4.entities))
        objects.append(segs4)

        segs5 = np.array([[a, d]])
        segs5 = trimesh.load_path(segs5)
        segs5.colors = np.array([[128, 128, 0]] * len(segs5.entities))
        objects.append(segs5)


        # o1 = pos + dir * size *10
        o1 = pos + dir * size * 30

        segs2 = np.array([[pos, o1]])
        segs2 = trimesh.load_path(segs2)
        segs2.colors = np.array([[255, 0, 0]] * len(segs2.entities))
        # objects.append(segs2)
    if poses2 is not None:
        for pose in poses2:
            # a camera is visualized with 8 line segments.
            pos = pose[:3, 3]
            a = pos - size * pose[:3, 0] - size * pose[:3, 1] + flag * size * pose[:3, 2]
            b = pos + size * pose[:3, 0] - size * pose[:3, 1] + flag * size * pose[:3, 2]
            c = pos + size * pose[:3, 0] + size * pose[:3, 1] + flag * size * pose[:3, 2]
            d = pos - size * pose[:3, 0] + size * pose[:3, 1] + flag * size * pose[:3, 2]

            dir = (a + b + c + d) / 4 - pos
            dir = dir / (np.linalg.norm(dir) + 1e-8)

            segs1 = np.array([[pos, a], [pos, b], [a, b]])
            segs1 = trimesh.load_path(segs1)
            segs1.colors = np.array([[200, 0, 0]] * len(segs1.entities))
            objects.append(segs1)
            segs3 = np.array([[pos, c], [pos, d], [c, d]])
            segs3 = trimesh.load_path(segs3)
            segs3.colors = np.array([[0, 200, 0]] * len(segs3.entities))
            objects.append(segs3)
            segs4 = np.array([[b, c]])
            segs4 = trimesh.load_path(segs4)
            segs4.colors = np.array([[0, 0, 200]] * len(segs4.entities))
            objects.append(segs4)

            segs5 = np.array([[a, d]])
            segs5 = trimesh.load_path(segs5)
            segs5.colors = np.array([[200, 200, 0]] * len(segs5.entities))
            objects.append(segs5)

            # o1 = pos + dir * size *10
            o1 = pos + dir * size * 30

            segs2 = np.array([[pos, o1]])
            segs2 = trimesh.load_path(segs2)
            segs2.colors = np.array([[0, 255, 0]] * len(segs2.entities))
            # objects.append(segs2)
    del pose
    trimesh.Scene(objects).show()

def visualize_poses_backup(poses1, poses2=None, size=0.01):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses1:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        # a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        # b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        # c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        # d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        a = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        b = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        c = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        d = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)



        segs1 = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs1 = trimesh.load_path(segs1)
        segs1.colors = np.array([[128, 0, 0]] * len(segs1.entities))
        objects.append(segs1)

        # o1 = pos + dir * size *10
        o1 = pos + dir * size * 30

        segs2 = np.array([[pos, o1]])
        segs2 = trimesh.load_path(segs2)
        segs2.colors = np.array([[150, 150, 150]] * len(segs2.entities))
        objects.append(segs2)
    if poses2 is not None:
        for pose in poses2:
            # a camera is visualized with 8 line segments.
            pos = pose[:3, 3]
            a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
            b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
            c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
            d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

            dir = pos - (a + b + c + d) / 4
            dir = dir / (np.linalg.norm(dir) + 1e-8)



            segs1 = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
            segs1 = trimesh.load_path(segs1)
            segs1.colors = np.array([[0, 128, 0]] * len(segs1.entities))
            objects.append(segs1)

            # o1 = pos + dir * size *10
            o1 = pos + dir * size * 30

            segs2 = np.array([[pos, o1]])
            segs2 = trimesh.load_path(segs2)
            segs2.colors = np.array([[0, 0, 0]] * len(segs2.entities))
            objects.append(segs2)
    del pose
    trimesh.Scene(objects).show()

def load_poses(dataset_path):
    dataset_path = Path(dataset_path)
    metadata_paths = sorted(list((dataset_path / 'train' / 'metadata').iterdir()))
    poses_mega = []
    intrinsics = []
    for x in metadata_paths:
        metadata = torch.load(x, map_location='cpu')
        pose = np.array(metadata['c2w'])
        intrinsic = np.array(
            [metadata['W'], metadata['H'],
             metadata['intrinsics'][0],
             metadata['intrinsics'][1],
             metadata['intrinsics'][2],
             metadata['intrinsics'][3]])
        intrinsics.append(intrinsic)
        poses_mega.append(pose)
    a=np.array(poses_mega)
    
    camera_positions = a[:,1:3,3]
    min_position = np.min(camera_positions,axis=0)
    max_position = np.max(camera_positions,axis=0)
    bianchang = (max_position- min_position) /8
    min_bound=min_position/8 - bianchang
    max_bound=max_position/8 - bianchang

    center_camera =np.where(
        (camera_positions[:, 0] >= min_bound[0]) & (camera_positions[:, 0] <= max_bound[0]) &
        (camera_positions[:, 1] >= min_bound[1]) & (camera_positions[:, 1] <= max_bound[1])
    )
    center_camera=list(center_camera)

    selected_poses = [poses_mega[int(index)] for index in center_camera[0]]
    # return poses_mega
    return selected_poses, center_camera

@record
@torch.inference_mode()




def main() -> None:
    if True:
        # pose_dji = load_poses(dataset_path='/disk1/Datasets/dji/DJI-mega')
        selected_poses, center_camera = load_poses(dataset_path='/data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-pixsfm')
        # visualize_poses(pose_dji_xml) #,pose_dji_xml)
        


if __name__ == '__main__':
    main()
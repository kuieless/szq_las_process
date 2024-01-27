import torch
import argparse
import collections
import os
import struct
from argparse import Namespace
from pathlib import Path

import trimesh
import pytorch3d
from pytorch3d.io import load_objs_as_meshes, load_obj
# import cv2
import numpy as np



def read_pose_after_single(path):
    metadata = torch.load(path)
    return metadata['c2w']

def read_pose_after(path):
    # pose_dir = [os.path.join(path, file_path) for file_path in os.listdir(path) if int(file_path.split('.')[-2])%10==0]
    pose_dir = [os.path.join(path, file_path) for file_path in os.listdir(path)]
    c2w_list = [torch.load(pose_file)['c2w'].numpy() for pose_file in pose_dir]
    return c2w_list


def visualize_poses_and_points(poses1, mesh=None, axis=None, size=0.1, flipz=True):
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
        segs3 = np.array([[pos, c], [pos, d], [c, d]])
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
        o1 = pos + dir * size * 5
        segs2 = np.array([[pos, o1]])
        segs2 = trimesh.load_path(segs2)
        segs2.colors = np.array([[255, 0, 0]] * len(segs2.entities))
        objects.append(segs2)

        segX = np.array([[(0,0,0), (1,0,0)]])
        segX = trimesh.load_path(segX)
        segX.colors = np.array([[255, 0, 0]] * len(segX.entities))
        objects.append(segX)
    if mesh is not None:
        objects.append(mesh)
    if axis is not None:
        for i, ax in enumerate(axis):
            line = np.array([[np.zeros(3), ax]])
            line = trimesh.load_path(line)
            if i==0:
                line.colors = np.array([[200, 0, 0]])
            elif i==1:
                line.colors = np.array([[0, 200, 0]])
            else:
                line.colors = np.array([[0, 0, 200]])
            objects.append(line)
    trimesh.Scene(objects).show()


def main():
    pose_after_path = r"C:\Users\87242\rebuttal_6371\IB\metadata"

    mesh_after_path = r"C:\Users\87242\rebuttal_6371\IB\norm.obj"

    pose_after = read_pose_after(pose_after_path)

    print("loading mesh")
    mesh_after = trimesh.load_mesh(mesh_after_path)

    visualize_poses_and_points(pose_after, mesh=mesh_after, axis=None)


if __name__ == '__main__':
    main()
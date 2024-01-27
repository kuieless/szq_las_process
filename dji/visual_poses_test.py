from pathlib import Path
import numpy as np
import torch
from torch.distributed.elastic.multiprocessing.errors import record
import trimesh
from scripts.colmap_to_mega_nerf import * #read_model, qvec2rotmat, RDF_TO_DRB

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_poses(poses1, poses2=None, size=0.01, flipz=True, point=None):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]
    
    flag = -1 if flipz else 1
    # poses1 = poses1[::10]
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
        objects.append(segs2)
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
    if point is not None:
        point = point[::100]
        points = trimesh.points.PointCloud(vertices=point)
        objects.append(points)
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
    return poses_mega

@record
@torch.inference_mode()




def main() -> None:
    if True:
        #
        # mesh = trimesh.load_mesh('/data/delete/residence_meshlabnorm.obj')
        # pts_3d = mesh.vertices
        # # pose_dji = load_poses(dataset_path='/disk1/Datasets/dji/DJI-mega')
        # pose_dji_xml = load_poses(dataset_path='/data/delete')
        # visualize_poses(pose_dji_xml, point=pts_3d) #,pose_dji_xml)

        # pose_rubble = load_poses(dataset_path='/disk1/Datasets/MegaNeRF/Mill19/rubble/rubble-pixsfm')
        # visualize_poses(pose_rubble)


        # pose_dji = load_poses(dataset_path='/disk1/Datasets/dji/DJI-mega')
        pose_dji_xml = load_poses(dataset_path='/data/delete')
        visualize_poses(pose_dji_xml) #,pose_dji_xml)

    else:
        cameras, images, _ = read_model('/disk1/yuqi/code/nerfstudio/output_dji/colmap/sparse/0')
        c2ws = {}
        for image in images.values():
            w2c = torch.eye(4)
            w2c[:3, :3] = torch.FloatTensor(qvec2rotmat(image.qvec))
            w2c[:3, 3] = torch.FloatTensor(image.tvec)
            c2w = torch.inverse(w2c)

            c2w = torch.hstack((
                RDF_TO_DRB @ c2w[:3, :3] @ torch.inverse(RDF_TO_DRB),
                RDF_TO_DRB @ c2w[:3, 3:]
            ))
            ZYQ = torch.FloatTensor([[0, 0, -1],
                                     [0, 1,0],
                                     [1, 0, 0]])
            c2w = torch.hstack((
                ZYQ @ c2w[:3, :3], #@ torch.inverse(ZYQ),
                ZYQ @ c2w[:3, 3:]
            ))

            c2ws[image.id] = c2w

        positions = torch.cat([c2w[:3, 3].unsqueeze(0) for c2w in c2ws.values()])
        print('{} images'.format(positions.shape[0]))
        max_values = positions.max(0)[0]
        min_values = positions.min(0)[0]
        origin = ((max_values + min_values) * 0.5)
        dist = (positions - origin).norm(dim=-1)
        diagonal = dist.max()
        scale = diagonal
        print(origin, diagonal, max_values, min_values)

        poses_process = []
        for i, image in enumerate(tqdm(sorted(images.values(), key=lambda x: x.name))):
            camera_in_drb = c2ws[image.id]
            camera_in_drb[:, 3] = (camera_in_drb[:, 3] - origin) / scale

            assert np.logical_and(camera_in_drb >= -1, camera_in_drb <= 1).all()
            pose_process = torch.cat([camera_in_drb[:, 1:2], -camera_in_drb[:, :1], camera_in_drb[:, 2:4]],-1)
            # pose_process = camera_in_drb

            # ZYQ = torch.FloatTensor([[0, 0, -1],
            #                          [0, 1,0],
            #                          [1, 0, 0]])
            # pose_process = torch.hstack((
            #     ZYQ @ pose_process[:3, :3], #@ torch.inverse(ZYQ),
            #     ZYQ @ pose_process[:3, 3:]
            # ))
            poses_process.append(pose_process.numpy())
        visualize_poses(poses_process)

if __name__ == '__main__':
    main()
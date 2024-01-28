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

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return R,t,S1_hat


def _get_mask_opts() -> Namespace:

    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--dataset_path', default='/disk1/Datasets/MegaNeRF/Mill19/rubble/rubble-pixsfm',type=str, required=False)
    parser.add_argument('--output', default='/disk1/yuqi/code/mega-nerf-zyq/render_images/rubble-custom',type=str, required=False)
    parser.add_argument('--resume', default=True, action='store_true')
    # parser.add_argument('--resume', default=True, action='store_false')
    parser.add_argument('--render_type', default='circle', type=str, help= 'spiral  or  right  or  circle')
    parser.add_argument('--circle_scale', default=0.5, type=float, help= 'adjust the size of circle according to the distribution of cameras')
    return parser.parse_known_args()[0]

def visualize_poses(poses1, poses2, size=0.01):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses1:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

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

    for pose in poses2:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
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

import open3d as o3d
import copy
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])



@record
@torch.inference_mode()
def main(hparams: Namespace) -> None:
    #render_type = hparams.render_type

    #from colmap
    # colmap_poses = open('/disk1/yuqi/code/nerfstudio/output_rubble/colmap/sparse/0/cameras.bin', 'rb')

    with open('/disk1/yuqi/code/nerfstudio/output_rubble/transforms.json', 'r', encoding='utf8') as fp:
        transform_json = json.load(fp)
    colmap_input = transform_json['frames']

    order = sorted(range(len(colmap_input)), key=lambda index: int(colmap_input[index]['file_path'][-9:-4]))
    colmap_poses = [colmap_input[index] for index in order]


    colmap_source = np.array([np.array(colmap_poses[i]['transform_matrix'])[0:3,3] for i in range(len(colmap_poses))])

    output_path = Path(hparams.output)
    output_path.mkdir(parents=True, exist_ok=hparams.resume)
    dataset_path = Path(hparams.dataset_path)
    # metadata_paths = sorted(list((dataset_path / 'train' / 'metadata').iterdir()) + list((dataset_path / 'val' / 'metadata').iterdir()))
    metadata_paths = sorted(list((dataset_path / 'train' / 'metadata').iterdir()))


    meta_target = np.array([torch.load(metadata_paths[i], map_location='cpu')['c2w'].numpy()[0:3,3] for i in range(len(metadata_paths))])


    max_values = colmap_source.max(0)
    min_values = colmap_source.min(0)
    origin = ((max_values + min_values) * 0.5)

    distance1 = np.linalg.norm(colmap_source[990] - colmap_source[0])
    distance2 = np.linalg.norm(meta_target[990] - meta_target[0])
    scale = distance1/distance2

    colmap_source = (colmap_source - origin) / scale




    point1 = colmap_source
    point2 = meta_target

    R, t, _ = compute_similarity_transform(point1, point2)
    '''icp'''
    import open3d as o3d
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(point2)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(point1)
    # draw_registration_result(source, target, np.identity(4))
    # trans_init = np.array([[1., 0., 0., 0.,],[0., 1., 0., 0.],[0., 0., 1., 0.],[0., 0., 0., 1.]])
    icp = o3d.pipelines.registration.registration_icp(source,target,1, np.identity(4),o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(icp.transformation)
    a= icp.transformation
    b= np.concatenate([R, t], 1)
    b= np.concatenate([b,a[3:,:]],0)
    # draw_registration_result(source, target, icp.transformation)
    # draw_registration_result(source, target, b)

    transform_matrix = torch.tensor(icp.transformation[0:3,:],dtype=torch.float32)



    poses111 = []
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
        poses111.append(pose)
    #
    # positions = torch.cat([c2w[:3, 3].unsqueeze(0) for c2w in c2ws.values()])
    # print('{} images'.format(positions.shape[0]))
    # max_values = positions.max(0)[0]
    # min_values = positions.min(0)[0]
    # origin = ((max_values + min_values) * 0.5)

    embeddings = torch.zeros(len(intrinsics), 1, dtype=torch.int)
    with open('/disk1/yuqi/code/mega-nerf-zyq/render_images/rubble.json', 'r', encoding='utf8') as fp:
        json_data = json.load(fp)


    RDF_TO_DRB = torch.FloatTensor([[0, 1, 0],
                                    [1, 0, 0],
                                    [0, 0, -1]])
    DRB_TO_RUB = torch.FloatTensor([[0, -1, 0],
                                    [1, 0, 0],
                                    [0, 0, -1]])

    # RDF_TO_DRB = torch.FloatTensor([[-1, 0, 0],
    #                                 [0, 1, 0],
    #                                 [0, 0, -1]])

    poses = []
    for i in range(len(json_data['camera_path'])):
        pose = np.array(json_data['camera_path'][i]['camera_to_world'])
        pose = pose[0:12].reshape(3,4)
        pose = torch.tensor(pose,dtype=torch.float32)




        #  studio -> colmap???
        pose[2,:] *= -1
        pose = pose[np.array([1,0,2]), :]
        pose[0:3, 1:3] *= -1
        #
        #
        #
        # pose[:, 3] = (pose[:, 3] - origin) / scale
        #
        # R = torch.tensor(R,dtype=torch.float32)
        # t = torch.tensor(t,dtype=torch.float32)
        # pose[:,3] = R[:3,:3] @ pose[:,3] + t.squeeze(-1)

        # pose[:, 3] = pose[:, 3]*scale + origin

        c2w = pose


        poses.append(np.array(c2w).reshape(3,4))

    nerf_studio = o3d.geometry.PointCloud()
    nerf_studio.points = o3d.utility.Vector3dVector(np.array(poses)[:,:,3])
    draw_registration_result(source, nerf_studio, np.identity(4))


    visualize_poses(poses, poses111[0:3])


    poses = torch.tensor(np.array(poses))


    max_values = poses[:,:3, 3].max(0)[0]
    min_values = poses[:,:3, 3].min(0)[0]
    origin = ((max_values + min_values) * 0.5)
    poses[:, :, 3] = (poses[:, :, 3] - origin)

    poses = np.array(poses).reshape(-1, 12)

    poses_file = hparams.output + '/poses.txt'
    intrinsics_file = hparams.output + '/intrinsics.txt'
    embeddings_file = hparams.output + '/embeddings.txt'

    np.savetxt(poses_file, poses,
               fmt='%.04f  %.04f  %.04f  %.04f  %.04f  %.04f  %.04f  %.04f  %.04f  %.04f  %.04f  %.04f')
    np.savetxt(intrinsics_file, intrinsics, fmt='%i  %i  %.04f  %.04f  %i  %i')
    np.savetxt(embeddings_file, embeddings, fmt='%i')

    print(poses_file)
    print('Done')

if __name__ == '__main__':
    main(_get_mask_opts())




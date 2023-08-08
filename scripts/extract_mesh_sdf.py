import sys

import torch
import mcubes
import numpy as np
from gp_nerf.opts import get_opts_base
import open3d as o3d

import os 
import trimesh


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 256
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    #with torch.no_grad():
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs) # for torch < 1.10, should remove indexing='ij'
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [1, N, 3]
                val = query_func(pts).reshape(len(xs), len(ys), len(zs)) # [1, N, 1] --> [x, y, z]
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val.detach().cpu().numpy()
                del val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, use_sdf = False):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    if use_sdf:
        u = - 1.0 *u

    #print(u.mean(), u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def save_mesh(nerf, device, save_path=None, resolution= 256, bound=1, threshold=0.,  use_sdf = False):
        if save_path is None:
            save_path = os.path.join("workspace", 'meshes', '1.obj')

        print(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    # sdfs, _ = nerf.density(pts.to(device), bound)
                    sdf_nn_output= nerf.forward_sdf(pts.to(device))
                    sdfs = sdf_nn_output[:, :1]
            return sdfs

        # w/ aabb
        # bounds_min = torch.FloatTensor([-bound] * 3)
        # bounds_max = torch.FloatTensor([bound] * 3)
        bounds_min = torch.FloatTensor([0.05, -0.1590, -0.3044])
        bounds_max = torch.FloatTensor([0.1051, 0.1580, 0.3046])

        vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=threshold, query_func = query_func, use_sdf = use_sdf)
        
        # align camera
        vertices = np.concatenate([vertices[:,2:3], vertices[:,0:1], vertices[:,1:2]], axis=-1)
        # vertices = (vertices - offset.numpy()) / scale
  
        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        print(f"==> Finished saving mesh.")


def _get_train_opts():
    parser = get_opts_base()
    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--dataset_path', type=str, required=True)
    # parser.add_argument('--threshold', type=int, default=100, required=False)

    return parser.parse_args()


def main(hparams) -> None:

    assert hparams.ckpt_path is not None or hparams.container_path is not None
    from gp_nerf.runner_gpnerf import Runner

    runner = Runner(hparams)
    nerf = runner.nerf
    nerf.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_mesh(nerf, device, save_path='./output/sci-art.obj', resolution=512, bound = hparams.aabb_bound, threshold=0, use_sdf=True)


if __name__ == '__main__':
    main(_get_train_opts())





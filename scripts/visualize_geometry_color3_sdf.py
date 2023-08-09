import sys

import torch
import mcubes
import numpy as np
from mega_nerf.opts import get_opts_base
from mega_nerf.datasets.dataset_utils import get_rgb_index_mask
from mega_nerf.ray_utils import get_rays, get_ray_directions
from torch.cuda.amp import GradScaler
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import open3d as o3d
from mega_nerf.misc_utils import main_print, main_tqdm

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


def save_mesh(nerf, device, save_path=None, resolution= 256, aabb = None, bound=1, threshold=0.,  use_sdf = False):
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
        
        # scale = max(0.000001,
        #             max(max(abs(float(aabb[1][0])-float(aabb[0][0])),
        #                     abs(float(aabb[1][1])-float(aabb[0][1]))),
        #                     abs(float(aabb[1][2])-float(aabb[0][2]))))
                        
        # scale = 2.0 * bound / scale

        # offset =  torch.FloatTensor([
        #             ((float(aabb[1][0]) + float(aabb[0][0])) * 0.5) * -scale,
        #             ((float(aabb[1][1]) + float(aabb[0][1])) * 0.5) * -scale, 
        #             ((float(aabb[1][2]) + float(aabb[0][2])) * 0.5) * -scale])


        # #demo setting w/o aabb
        # bounds_min = torch.FloatTensor([-0.4 * bound, -0.8 *bound, -0.3 *bound])
        # bounds_max = torch.FloatTensor([0.4 *bound, 0.4 * bound, 0.3 *bound])
        
        #ficus setting w/o aabb
        # bounds_min = torch.FloatTensor([-0.35 * bound, -1.0 *bound, -0.35 *bound])
        # bounds_max = torch.FloatTensor([0.35 *bound, 0.4 * bound, 0.35 *bound])

        # w/ aabb
        bounds_min = torch.FloatTensor([-bound] * 3)
        bounds_max = torch.FloatTensor([bound] * 3)

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
    from mega_nerf.runner_sdf import Runner
    if hparams.zyq_code:
        print("run clean version, remove the bg nerf")
        hparams.bg_nerf = False
    else:
        if hparams.mega_with == 'None':
            print("run original Mega_NeRF")
            hparams.layer_dim = 256
            hparams.coarse_samples = 256
            hparams.fine_samples = 512


    runner = Runner(hparams)
    nerf = runner.nerf
    nerf.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if hparams.ckpt_path is not None:
    #     state_dict = torch.load(hparams.ckpt_path, map_location='cpu')['model_state_dict']
    #     consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')
    #     model_dict = nerf.state_dict()
    #     model_dict.update(state_dict)
    #     nerf.load_state_dict(model_dict)

    save_path = './mesh/sci-art.obj'
    
    sphere_center = runner.sphere_center
    sphere_radius = runner.sphere_radius
    aabb = torch.tensor([
        [sphere_center[0]-sphere_radius[0], sphere_center[1]-sphere_radius[1], sphere_center[2]-sphere_radius[2]],
        [sphere_center[0]+sphere_radius[0], sphere_center[1]+sphere_radius[1], sphere_center[2]+sphere_radius[2]],
    ],device=device)
    save_mesh(nerf, device, save_path=save_path, resolution=512, aabb=aabb, bound = hparams.aabb_bound, threshold=0, use_sdf=True)






    # N = 1024  # 512

    # bound = 1
    # full_mesh = False  # True False
    # if full_mesh:
    #     t = np.linspace(-bound, bound, N + 1)
    #     query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
    # else:
    #     t = np.linspace(0, 1, N + 1)
    #     # t1 = np.linspace(-0.5, 0.5, 5 * N + 1)
    #     t1 = np.linspace(0, 1, int(N/5) + 1)
    #     query_pts = np.stack(np.meshgrid(t1, t, t), -1).astype(np.float32)

    # sh = query_pts.shape
    # flat = query_pts.reshape([-1, 3])
    # del query_pts
    # chunk = 1024*4
    # out_chunks = []
    # points = []
    # colors = []
    # # get dir
    # metadata_item = runner.train_items[0]
    # image_data = get_rgb_index_mask(metadata_item)

    # if image_data is None:
    #     print("load data error")
    #     sys.exit()
    # image_rgbs, image_indices, image_keep_mask = image_data
    # # print("image index: {}, fx: {}, fy: {}".format(metadata_item.image_index, metadata_item.intrinsics[0], metadata_item.intrinsics[1]))
    # directions = get_ray_directions(metadata_item.W,
    #                                 metadata_item.H,
    #                                 metadata_item.intrinsics[0],
    #                                 metadata_item.intrinsics[1],
    #                                 metadata_item.intrinsics[2],
    #                                 metadata_item.intrinsics[3],
    #                                 False,
    #                                 device)
    # near = 0
    # far = 2
    # ray_altitude_range = [11, 38]
    # image_rays = get_rays(directions, metadata_item.c2w.to(device), near, far, ray_altitude_range).view(-1,8).cpu()

    # dir = image_rays[100,3:6]
    # index = torch.tensor(1,dtype = torch.int)
    # del image_rays, metadata_item, image_data, directions
    # for i in main_tqdm(range(0, flat.shape[0], chunk)):
    #     xyz_chunk = torch.tensor(flat[i: i + chunk]).cuda()
    #     dir_chunk = dir.repeat(xyz_chunk.size(0), 1).cuda()
    #     index_chunk = index.repeat(xyz_chunk.size(0), 1).cuda()
    #     input_chunk = torch.cat([xyz_chunk, dir_chunk, index_chunk], dim=1)
    #     del dir_chunk, index_chunk, xyz_chunk
    #     model_chunk = nerf('fg', input_chunk, sigma_noise=None)
    #     flat_chunk = flat[i: i + chunk]
    #     model_chunk = model_chunk.detach().cpu()
    #     visual_point_index = (model_chunk[:,-1] > threshold).reshape([-1, 1]).squeeze(1)
    #     visual_point = flat_chunk[visual_point_index]
    #     color = (model_chunk[..., :-1].reshape([-1, 3]))[visual_point_index]
    #     points += [visual_point]
    #     colors += [color]
    #     del input_chunk, model_chunk,flat_chunk, visual_point_index, visual_point, color

    #     # out_chunks += [model_chunk.detach().cpu()]
    #     # print(i)
    # # out = torch.cat(out_chunks, 0)
    # # del out_chunks
    # # out = np.reshape(out.detach().cpu().numpy(), list(sh[:-1]) + [-1])
    # # # visual_point_index = (out[..., -1] > threshold).reshape([-1, 1]).squeeze(1)
    # # visual_point = flat[visual_point_index]
    # # color = (out[..., :-1].reshape([-1, 3]))[visual_point_index]
    # # del out
    # visual_point = np.concatenate(points, 0)
    # visual_color = np.concatenate(colors, 0)
    # points = np.asarray(visual_point) / hparams.bound
    # colors = np.asarray(visual_color)

    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(points)
    # pcd2.colors = o3d.utility.Vector3dVector(colors)
    # o3d.io.write_point_cloud("./data_geometry.ply", pcd2)
    # print("write the ply files")
    # o3d.visualization.draw_geometries([pcd2])
    # # o3d.visualization.draw_geometries([pcd2],
    # #                                   zoom=0.3412,
    # #                                   front=[0.4257, -0.2125, -0.8795],
    # #                                   lookat=[2.6172, 2.0475, 1.532],
    # #                                   up=[-0.0694, -0.9768, 0.2024])
    # a=1



if __name__ == '__main__':
    main(_get_train_opts())





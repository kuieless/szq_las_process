import sys

import torch
import mcubes
import numpy as np
from gp_nerf.opts import get_opts_base
from mega_nerf.ray_utils import get_rays, get_ray_directions
from torch.cuda.amp import GradScaler
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import open3d as o3d
from mega_nerf.misc_utils import main_print, main_tqdm



def _get_train_opts():
    parser = get_opts_base()
    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--threshold', type=int, default=50, required=False)

    return parser.parse_args()


def main(hparams) -> None:
    threshold = hparams.threshold
    
    from gp_nerf.runner_gpnerf import Runner
    hparams.bg_nerf= False
    runner = Runner(hparams)
    scaler = torch.cuda.amp.GradScaler(enabled=hparams.amp)
    nerf = runner.nerf
    nerf.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert hparams.ckpt_path is not None

    state_dict = torch.load(hparams.ckpt_path, map_location='cpu')['model_state_dict']
    consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')
    model_dict = nerf.state_dict()
    model_dict.update(state_dict)
    nerf.load_state_dict(model_dict)

    
    N = 512  # 512

    # bound = runner.hparams.aabb_bound.cpu().numpy()
    bound = 0.7
    full_mesh = False  # True False
    if full_mesh:
        t = np.linspace(-bound, bound, N + 1)
        query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
    else:
        # bound的计算在dji/project_script/6_process_depth2nerf_ds_save_0801night.py里
        # t1 = np.linspace(0.0055, 0.1211, int(N/2) + 1)
        # t2 = np.linspace(-0.2990, 0.3280, N + 1)
        # t3 = np.linspace(-0.6344, 0.6546, N + 1)

        t1 = np.linspace(0.05, 0.1051, int(N/3) + 1)
        t2 = np.linspace(-0.1590, 0.1580, N + 1)
        t3 = np.linspace(-0.3044, 0.3046, N + 1)

        # t = np.linspace(0, 1, N + 1)
        # t1 = np.linspace(0, 1, int(N/5) + 1)

        query_pts = np.stack(np.meshgrid(t1, t2, t3), -1).astype(np.float32)


    sh = query_pts.shape
    flat = query_pts.reshape([-1, 3])
    del query_pts
    chunk = 1024*4
    out_chunks = []
    points = []
    colors = []
    # get dir
    metadata_item = runner.train_items[0]


    directions = get_ray_directions(metadata_item.W,
                                    metadata_item.H,
                                    metadata_item.intrinsics[0],
                                    metadata_item.intrinsics[1],
                                    metadata_item.intrinsics[2],
                                    metadata_item.intrinsics[3],
                                    False,
                                    device)
    near = 0
    far = 2
    ray_altitude_range = hparams.ray_altitude_range
    image_rays = get_rays(directions, metadata_item.c2w.to(device), near, far, ray_altitude_range).view(-1,8).cpu()

    dir = image_rays[100,3:6]
    index = torch.tensor(1,dtype = torch.int)
    del image_rays, metadata_item, directions
    for i in main_tqdm(range(0, flat.shape[0], chunk)):
        xyz_chunk = torch.tensor(flat[i: i + chunk]).cuda()
        dir_chunk = dir.repeat(xyz_chunk.size(0), 1).cuda()
        index_chunk = index.repeat(xyz_chunk.size(0), 1).cuda()
        input_chunk = torch.cat([xyz_chunk, dir_chunk, index_chunk], dim=1)
        del dir_chunk, index_chunk, xyz_chunk
        model_chunk = nerf('fg', input_chunk, sigma_noise=None)
        flat_chunk = flat[i: i + chunk]
        model_chunk = model_chunk.detach().cpu()
        visual_point_index = (model_chunk[:,-1] > threshold).reshape([-1, 1]).squeeze(1)
        visual_point = flat_chunk[visual_point_index]
        color = (model_chunk[..., :-1].reshape([-1, 3]))[visual_point_index]
        points += [visual_point]
        colors += [color]
        del input_chunk, model_chunk,flat_chunk, visual_point_index, visual_point, color

        # out_chunks += [model_chunk.detach().cpu()]
        # print(i)
    # out = torch.cat(out_chunks, 0)
    # del out_chunks
    # out = np.reshape(out.detach().cpu().numpy(), list(sh[:-1]) + [-1])
    # # visual_point_index = (out[..., -1] > threshold).reshape([-1, 1]).squeeze(1)
    # visual_point = flat[visual_point_index]
    # color = (out[..., :-1].reshape([-1, 3]))[visual_point_index]
    # del out
    visual_point = np.concatenate(points, 0)
    visual_color = np.concatenate(colors, 0)
    points = np.asarray(visual_point) / hparams.aabb_bound.cpu().numpy()
    colors = np.asarray(visual_color)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points)
    pcd2.colors = o3d.utility.Vector3dVector(colors)

    name = hparams.ckpt_path.split('/')
    model = (name[-1].split('.'))[0]
    save_path = f'./output/{name[1]}_{model}_t{threshold}.ply'
    o3d.io.write_point_cloud(f"{save_path}", pcd2)
    print(f"write the ply files at {save_path}.")

    
    # o3d.visualization.draw_geometries([pcd2])
    # o3d.visualization.draw_geometries([pcd2],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])
    # a=1



if __name__ == '__main__':
    main(_get_train_opts())





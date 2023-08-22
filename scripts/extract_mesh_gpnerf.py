import sys

import torch
import mcubes
import numpy as np
from mega_nerf.ray_utils import get_rays, get_ray_directions
from torch.cuda.amp import GradScaler
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import open3d as o3d
from mega_nerf.misc_utils import main_print, main_tqdm
import trimesh

from gp_nerf.opts import get_opts_base

def _get_train_opts():
    parser = get_opts_base()
    # parser.add_argument('--threshold', type=int, default=2000, required=False)
    parser.add_argument('--threshold', type=int, default=50, required=False)
    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--dataset_path', type=str, required=True)
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


    N = 256  # 512

    # bound = runner.hparams.aabb_bound.cpu().numpy()
    bound = 0.5

    full_mesh = False  # True False
    if full_mesh:
        t = np.linspace(-bound, bound, N + 1)
        query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
    else:
        # bound的计算在dji/project_script/6_process_depth2nerf_ds_save_0801night.py里
        # t1 = np.linspace(0.0055, 0.1211, int(N/2) + 1)
        # t2 = np.linspace(-0.2990, 0.3280, N + 1)
        # t3 = np.linspace(-0.6344, 0.6546, N + 1)
        ####  dji xiayuan的设置
        # t1a, t1b = 0.05, 0.1051
        # t2a, t2b = -0.1590, 0.1580
        # t3a, t3b = -0.3044, 0.3046
        ##residence subset
        t1a, t1b = 0.2, 1
        t2a, t2b = -1, 1
        t3a, t3b = -1, 1

        t1 = np.linspace(t1a, t1b, N + 1)
        t2 = np.linspace(t2a, t2b, N + 1)
        t3 = np.linspace(t3a, t3b, N + 1)


        query_pts = np.stack(np.meshgrid(t1, t2, t3), -1).astype(np.float32)

    sh = query_pts.shape
    flat = query_pts.reshape([-1, 3])
    # del query_pts
    chunk = 1024*4
    out_chunks = []
    points = []
    colors = []
    # get dir
    metadata_item = runner.train_items[0]

    # print("image index: {}, fx: {}, fy: {}".format(metadata_item.image_index, metadata_item.intrinsics[0], metadata_item.intrinsics[1]))
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
    ray_altitude_range = [11, 38]
    image_rays = get_rays(directions, metadata_item.c2w.to(device), near, far, ray_altitude_range).view(-1,8).cpu()

    dir = image_rays[100,3:6]
    index = torch.tensor(1,dtype = torch.int)
    del image_rays, metadata_item, directions


    out_chunks = []
    for i in main_tqdm(range(0, flat.shape[0], chunk)):
        xyz_chunk = torch.tensor(flat[i: i + chunk]).cuda()
        dir_chunk = dir.repeat(xyz_chunk.size(0), 1).cuda()
        index_chunk = index.repeat(xyz_chunk.size(0), 1).cuda()
        input_chunk = torch.cat([xyz_chunk, dir_chunk, index_chunk], dim=1)
        del dir_chunk, index_chunk, xyz_chunk
        model_chunk = nerf('fg', input_chunk)
        # flat_chunk = flat[i: i + chunk]
        # model_chunk = model_chunk.detach().cpu()

        out_chunks += [model_chunk[:,-1].cpu().detach().numpy()]
        del model_chunk, input_chunk
    out = np.concatenate(out_chunks, axis=0)
    # out = torch.cat(out_chunks, 0)
    # sigmas = out[..., -1]  
    sigmas = out.reshape(sh[:-1])
    # print(sigmas.shape)
    # print(flat.max(0))
    # print(flat.min(0))

    # (t1b-t1a, t2b-t2a, t3b-t3a)

    vertices, triangles = mcubes.marching_cubes(sigmas, threshold)
    # print(vertices.shape)
    # print(vertices.max(0))
    # print(vertices.min(0))
    
    vertices = vertices * np.array((t2b-t2a, t1b-t1a, t3b-t3a))

    mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
    
    name = hparams.ckpt_path.split('/')
    model = (name[-1].split('.'))[0]
    save_path = f'./output/{name[1]}_{model}_t{threshold}.obj'
    mesh.export(save_path)

    print(f"==> Finished saving mesh at {save_path}.")
        



if __name__ == '__main__':
    main(_get_train_opts())





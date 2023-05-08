from argparse import Namespace

import torch
from torch.distributed.elastic.multiprocessing.errors import record

import sys
sys.path.append('.')

from gp_nerf.opts import get_opts_base



def _get_eval_opts() -> Namespace:
    parser = get_opts_base()

    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--centroid_path', type=str)

    return parser.parse_args()

@record
def main(hparams: Namespace) -> None:
    assert hparams.ckpt_path is not None or hparams.container_path is not None
    if hparams.network_type == 'sdf':
        hparams.cos_iterations = int(hparams.train_iterations / 2)
        hparams.normal_iterations = int(hparams.train_iterations / 2)

    from gp_nerf.runner_gpnerf import Runner
    print(f"stop_semantic_grad:{hparams.stop_semantic_grad}")
    print(f"use_pano_lift:{hparams.use_pano_lift}")

    hparams.bg_nerf = False

    
    Runner(hparams)._run_validation_project_val_points()


if __name__ == '__main__':
    main(_get_eval_opts())

# --exp_name logs_357/0504_G3_geo_residence --enable_semantic False --pos_xyz_dim 10 --label_name m2f_new --separate_semantic True --network_type gpnerf --config_file configs/residence.yaml --dataset_path /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels --chunk_paths /data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence_chunk-labels-m2f_new-down4

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
    if hparams.network_type == 'sdf':
        # hparams.cos_iterations = int(hparams.train_iterations / 2)
        # hparams.normal_iterations = int(hparams.train_iterations / 2)
        pass
    elif hparams.network_type == 'sdf_mlp':
        hparams.use_neus_gradient=True
        hparams.layer_dim =256
    elif 'nr3d' in hparams.network_type:
        # hparams.log2_hashmap_size=20
        hparams.contract_new=True
        hparams.sdf_include_input=False
        
    from gp_nerf.runner_gpnerf import Runner
    print(f"stop_semantic_grad:{hparams.stop_semantic_grad}")
    print(f"use_pano_lift:{hparams.use_pano_lift}")

    hparams.bg_nerf = False


    Runner(hparams).val_3d_to_2d()


if __name__ == '__main__':
    main(_get_eval_opts())

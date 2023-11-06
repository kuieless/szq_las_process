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
    

    if 'Longhua' in hparams.dataset_path:
        hparams.train_scale_factor = 1
        hparams.val_scale_factor = 1
        
    if 'linear_assignment' in hparams.instance_loss_mode:
        assert hparams.num_instance_classes > 30
    else:
        hparams.num_instance_classes = 25
        assert hparams.num_instance_classes < 30
        
    from gp_nerf.runner_gpnerf import Runner
    print(f"stop_semantic_grad:{hparams.stop_semantic_grad}")
    print(f"use_pano_lift:{hparams.use_pano_lift}")

    hparams.bg_nerf = False

    if hparams.render_zyq:
        # hparams.val_scale_factor = 4
        # hparams.train_scale_factor = 4
        hparams.depth_dji_type = 'mesh'
        hparams.visual_normal=True

        if hparams.detect_anomalies:
            with torch.autograd.detect_anomaly():
                Runner(hparams).render_zyq()
        else:
            Runner(hparams).render_zyq()
    else:
        if hparams.detect_anomalies:
            with torch.autograd.detect_anomaly():
                Runner(hparams).eval()
        else:
            Runner(hparams).eval()


if __name__ == '__main__':
    main(_get_eval_opts())

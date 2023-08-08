import os
from argparse import Namespace
from typing import Optional, Dict, Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from mega_nerf.spherical_harmonics import eval_sh
from gp_nerf.sample_bg import bg_sample_inv, contract_to_unisphere

# TO_COMPOSITE = {'rgb', 'depth', 'sem_map'}
#sdf
TO_COMPOSITE = {'rgb', 'depth'}
INTERMEDIATE_KEYS = {'zvals_coarse', 'raw_rgb_coarse', 'raw_sigma_coarse', 'depth_real_coarse', 'raw_sem_logits_coarse', 'raw_sem_feature_coarse'}

def near_far_from_bound(rays_o, rays_d, bound, type='cube'):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound # [B, N, 1]
        far = radius + bound

    elif type == 'cube':
        # TODO: if bound < radius, some rays may not intersect with the bbox. (should set near = far = inf ... or far < near)
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        near = torch.clamp(near, min=0.05)

    return near, far


def render_rays(nerf: nn.Module,
                bg_nerf: Optional[nn.Module],
                rays: torch.Tensor,
                image_indices: Optional[torch.Tensor],
                hparams: Namespace,
                sphere_center: Optional[torch.Tensor],
                sphere_radius: Optional[torch.Tensor],
                get_depth: bool,
                get_depth_variance: bool,
                get_bg_fg_rgb: bool,
                train_iterations=-1,
                gt_depths=None,
                depth_scale=None) -> Tuple[Dict[str, torch.Tensor], bool]:
    N_rays = rays.shape[0]
    device = rays.device
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)
    near = torch.clamp(near, max=1e4-1)
    if image_indices is not None:
        image_indices = image_indices.unsqueeze(-1).unsqueeze(-1)

    perturb = hparams.perturb if nerf.training else 0
    last_delta = 1e10 * torch.ones(N_rays, 1, device=device)


    if True: #use mega's points-segmetation method with ellipsoid
        # 这边计算ray和椭圆的交点，从而得到fg_far
        # 之前的far已经根据altitude调整过一次
        # add by zyq 2023/02/23
        # sphere_center = torch.tensor([0, 0, 0], device=device)
        # sphere_radius = torch.tensor([hparams.aabb_bound, hparams.aabb_bound, hparams.aabb_bound],device=device)
        
        fg_far = _intersect_sphere(rays_o, rays_d, sphere_center, sphere_radius)
        fg_far = torch.maximum(fg_far, near.squeeze())
        # 划分bg ray
        rays_with_bg = torch.arange(N_rays, device=device)[far.squeeze() > fg_far]
        rays_with_fg = torch.arange(N_rays, device=device)[far.squeeze() <= fg_far]
    assert rays_with_bg.shape[0] + rays_with_fg.shape[0] == far.shape[0]
    rays_o = rays_o.view(rays_o.shape[0], 1, rays_o.shape[1])
    rays_d = rays_d.view(rays_d.shape[0], 1, rays_d.shape[1])
    if rays_with_bg.shape[0] > 0:
        last_delta[rays_with_bg, 0] = fg_far[rays_with_bg]

    #  zyq:    初始化
    far_ellipsoid = torch.minimum(far.squeeze(), fg_far).unsqueeze(-1)
    z_vals_inbound = torch.zeros([rays_o.shape[0], hparams.coarse_samples], device=device)
    # 属于fg的ray采样
    z_fg = torch.linspace(0, 1, hparams.coarse_samples, device=device)
    z_vals_inbound[rays_with_fg] = near[rays_with_fg] * (1 - z_fg) + far_ellipsoid[rays_with_fg] * z_fg
    # 属于bg的ray，其中fg部分的点采样
    z_bg_inner = torch.linspace(0, 1, hparams.coarse_samples, device=device)
    z_vals_inbound[rays_with_bg] = near[rays_with_bg] * (1 - z_bg_inner) + far_ellipsoid[rays_with_bg] * z_bg_inner
    #####随机扰动
    # z_vals_inbound = _expand_and_perturb_z_vals(z_vals_inbound, hparams.coarse_samples, perturb, N_rays)
    # xyz_coarse_fg = rays_o + rays_d * z_vals_inbound.unsqueeze(-1)

    # xyz_coarse_fg = contract_to_unisphere(xyz_coarse_fg, hparams)


    results = _get_results(point_type='fg',
                           nerf=nerf,
                           rays_o=rays_o,
                           rays_d=rays_d,
                           near=near,
                           far=far_ellipsoid,
                           image_indices=image_indices,
                           hparams=hparams,
                        #    xyz_coarse=xyz_coarse_fg,
                           z_vals=z_vals_inbound,
                           rays_with_bg=rays_with_bg,
                           last_delta=last_delta,
                           get_depth=get_depth,
                           get_depth_variance=get_depth_variance,
                           get_bg_lambda=True,
                           depth_real=None,
                           xyz_fine_fn=lambda fine_z_vals: (rays_o + rays_d * fine_z_vals.unsqueeze(-1), None),
                           train_iterations=train_iterations,
                           gt_depths=gt_depths,
                           depth_scale=gt_depths)
    
    if rays_with_bg.shape[0] != 0:
        z_vals_outer = bg_sample_inv(far_ellipsoid[rays_with_bg], 1e4+1, hparams.coarse_samples // 2, device)
        z_vals_outer = _expand_and_perturb_z_vals(z_vals_outer, hparams.coarse_samples // 2, perturb, rays_with_bg.shape[0])

        xyz_coarse_bg = rays_o[rays_with_bg] + rays_d[rays_with_bg] * z_vals_outer.unsqueeze(-1)
        xyz_coarse_bg = contract_to_unisphere(xyz_coarse_bg, hparams)

        bg_results = _get_results_bg(point_type='bg',
                                  nerf=nerf,
                                  rays_d=rays_d[rays_with_bg],
                                  image_indices=image_indices[rays_with_bg] if image_indices is not None else None,
                                  hparams=hparams,
                                  xyz_coarse=xyz_coarse_bg,
                                  z_vals=z_vals_outer,
                                  # bg_nerf的last_dalta为1e10
                                  last_delta=1e10 * torch.ones(rays_with_bg.shape[0], 1, device=device),
                                #   get_depth=get_depth,
                                  get_depth=True,
                                  get_depth_variance=get_depth_variance,
                                  get_bg_lambda=False,
                                  depth_real=None,
                                  xyz_fine_fn=lambda fine_z_vals: (rays_o[rays_with_bg] + rays_d[rays_with_bg] * fine_z_vals.unsqueeze(-1), None),
                                  train_iterations=train_iterations,
                                  gt_depths=gt_depths,
                                  depth_scale=gt_depths)
        
    # merge the result of inner and outer
    types = ['fine' if hparams.fine_samples > 0 else 'coarse']
    if hparams.use_cascade and hparams.fine_samples > 0:
        types.append('coarse')
    for typ in types:
        if rays_with_bg.shape[0] > 0:
            bg_lambda = results[f'bg_lambda_{typ}'][rays_with_bg]

            for key in TO_COMPOSITE:
                if f'{key}_{typ}' not in results:
                    continue

                val = results[f'{key}_{typ}']

                if get_bg_fg_rgb:
                    results[f'fg_{key}_{typ}'] = val

                expanded_bg_val = torch.zeros_like(val)

                mult = bg_lambda
                if len(val.shape) > 1:
                    mult = mult.unsqueeze(-1)


                if hparams.stop_semantic_grad and key == 'sem_map':
                    mult = mult.detach()
                expanded_bg_val[rays_with_bg] = bg_results[f'{key}_{typ}'] * mult

                if get_bg_fg_rgb:
                    results[f'bg_{key}_{typ}'] = expanded_bg_val
                results[f'{key}_{typ}'] = val + expanded_bg_val

        elif get_bg_fg_rgb:
            for key in TO_COMPOSITE:
                if f'{key}_{typ}' not in results:
                    continue

                val = results[f'{key}_{typ}']
                results[f'fg_{key}_{typ}'] = val
                results[f'bg_{key}_{typ}'] = torch.zeros_like(val)

    bg_nerf_rays_present = False

    return results, bg_nerf_rays_present


def _get_results(point_type,
                 nerf: nn.Module,
                 rays_o: torch.Tensor,
                 rays_d: torch.Tensor,
                 near,  # sdf
                 far,   # sdf
                 image_indices: Optional[torch.Tensor],
                 hparams: Namespace,
                #  xyz_coarse: torch.Tensor,
                 z_vals: torch.Tensor,   # z_vals_inbound
                 rays_with_bg,
                 last_delta: torch.Tensor,
                 get_depth: bool,
                 get_depth_variance: bool,
                 get_bg_lambda: bool,
                 depth_real: Optional[torch.Tensor],
                 xyz_fine_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]],
                 train_iterations=-1,
                 gt_depths=None,
                 depth_scale=None)-> Dict[str, torch.Tensor]:
    results = {}

    last_delta_diff = torch.zeros_like(last_delta)
    last_delta_diff[last_delta.squeeze() < 1e10, 0] = z_vals[last_delta.squeeze() < 1e10].max(dim=-1)[0]
    last_delta=last_delta - last_delta_diff

    device = rays_o.device

    #2023/02/22
    cos_anneal_ratio = min(train_iterations / (2*hparams.cos_iterations), 0.8)   #  from 0 -> 0.8
    normal_epsilon_ratio = min((train_iterations - hparams.normal_iterations) / (2*hparams.normal_iterations), 0.95)
    curvature_loss = False  #   和torch-nsr一样，暂时不考虑

    # ###### # ---------------gpnerf sampling and contraction-------------------
    # perturb z_vals
    bound = 0

    ####随机扰动
    perturb = hparams.perturb if nerf.training else 0
    z_vals = _expand_and_perturb_z_vals(z_vals, hparams.coarse_samples, perturb, rays_o.shape[0])
    xyz_coarse_fg = rays_o + rays_d * z_vals.unsqueeze(-1)
    pts = contract_to_unisphere(xyz_coarse_fg, hparams)
    
    # ##### ---------------same sampling as instant-nsr-------------------
    # bound=hparams.aabb_bound
    # rays_o = rays_o.view(-1, 3)
    # rays_d = rays_d.view(-1, 3)
    # # sample steps
    # near, far = near_far_from_bound(rays_o, rays_d, bound, type='cube')
    # # near, far = near_far_from_bound(rays_o, rays_d, bound, type='sphere')

    # z_vals = torch.linspace(0.0, 1.0, hparams.coarse_samples, device=device).unsqueeze(0)# [1, T]
    # z_vals = z_vals.expand((rays_o.shape[0], hparams.coarse_samples)) # [N, T]
    # z_vals = near + (far - near) * z_vals # [N, T], in [near, far]

    # # perturb z_vals
    # sample_dist = (far - near) / hparams.coarse_samples
    # if nerf.training:
    #     z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

    # # generate pts
    # pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 3] -> [N, T, 3]
    # pts = pts.clamp(-bound, bound) # must be strictly inside the bounds, else lead to nan in hashgrid encoder!

    # ######## ---------------end -------------------


    rays_o = rays_o.view(-1, 3)
    rays_d = rays_d.view(-1, 3)

    """ Step 1: Compute coarse sigma and weights"""
    typ='coarse'

    N_rays_ = pts.shape[0]
    N_samples_ = pts.shape[1]
    xyz_ = pts.view(-1, pts.shape[-1])
    


    if hparams.fine_samples > 0:  # sample points for fine model
        """" Step 2: Sample fine point """
        typ = 'fine'
        with torch.no_grad():
        # if True:
            # query SDF and RGB
            sdf_nn_output = nerf.forward_sdf(xyz_)
            sdf = sdf_nn_output[..., 0]
            sdf = sdf.view(N_rays_, N_samples_)
            if point_type == 'fg':
                fine_sample = hparams.fine_samples
            elif point_type == 'bg_same_as_fg':
                fine_sample = hparams.fine_samples // 2
            else:
                fine_sample = hparams.fine_samples // 2

            for i in range(fine_sample // 16):
                new_z_vals = up_sample(rays_o, rays_d, z_vals, sdf, 16, 64 * 2 **i)
                z_vals, sdf = cat_z_vals(nerf, rays_o, rays_d, z_vals, new_z_vals, sdf, bound, last=(i + 1 == fine_sample // 16))
    
    # ###这里完成z_vals（coarse+fine）
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # [N, T-1]
    deltas = torch.cat([deltas, last_delta], -1)  # (N_rays, N_samples_)

    # # sample pts on new z_vals, 取中点以及最后一个点
    z_vals_mid = (z_vals[:, :-1] + 0.5 * deltas[:, :-1]) # [N, T-1]
    z_vals_mid = torch.cat([z_vals_mid, z_vals[:,-1:]], dim=-1)
    new_pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals_mid.unsqueeze(-1) # [N, 1, 3] * [N, t, 3] -> [N, t, 3]
    
    # # ----instant-nsr or gpnerf---------  
    # NOTE：这里得到的采样点怎么处理
    # new_pts = new_pts.clamp(-bound, bound)
    new_pts = contract_to_unisphere(new_pts, hparams)

    
    """ Step 3: Compute fine sigma """
    N_rays_ = new_pts.shape[0]
    N_samples_ = new_pts.shape[1]   # 这里点的数量已经是coarse + fine
    xyz_ = new_pts.view(-1, new_pts.shape[-1])
    image_indices_ = image_indices.repeat(1, N_samples_, 1).view(-1, 1)
    B = xyz_.shape[0]
    
    # only forward new points to save computation
    rays_d_ = rays_d.unsqueeze(-2).repeat(1, N_samples_, 1).view(-1, rays_d.shape[-1])
    sdf_nn_output = nerf.forward_sdf(xyz_)

    if hparams.enable_semantic:
        sem_logits = nerf.forward_semantic(sdf_nn_output)
    
    sdf = sdf_nn_output[:, :1]
    feature_vector = sdf_nn_output[:, 1:]
    
    gradient = nerf.gradient(xyz_, 0.005 * (1.0 - normal_epsilon_ratio)).squeeze()

    # if nerf.training:
    #     gradient = nerf.gradient_neus(xyz_).squeeze()
    # else:
    #     with torch.enable_grad():
    #         gradient = nerf.gradient_neus(xyz_).squeeze()

    normal =  gradient / (1e-5 + torch.linalg.norm(gradient, ord=2, dim=-1,  keepdim = True))  # 这里instant-nsr 把gridient转换成了normal

    color = nerf.forward_color(xyz_, rays_d_, normal.reshape(-1, 3), feature_vector, image_indices_)

    # TODO: zyq - save the inv_s_ori parameter to wandb or tb
    inv_s_ori = nerf.forward_variance()     # Single parameter
    inv_s = inv_s_ori.expand(N_rays_ * N_samples_, 1)

    true_cos = (rays_d_.reshape(-1, 3) * normal).sum(-1, keepdim=True)  #[-1, 0]
    # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
    # the cos value "not dead" at the beginning training iterations, for better convergence.

    # version Softplus
    activation = nn.Softplus(beta=100)
    # anneal from 0 to 1
    # 第一项softplus([0, 1]) * (1 - anneal)
    # 第二项softplus([-1,0]) * anneal
    iter_cos = -(activation(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +  
                activation(-true_cos) * cos_anneal_ratio)  # always non-positive

    # add by zyq : change the last_delta to 1e10 for the fg points 2023/02/20/
    deltas[rays_with_bg] = torch.cat([deltas[rays_with_bg, :-1],  1e10 * torch.ones_like(deltas[rays_with_bg, :1])], dim=-1)
    

    # Estimate signed distances at section points
    estimated_next_sdf = sdf + iter_cos * deltas.reshape(-1, 1) * 0.5
    estimated_prev_sdf = sdf - iter_cos * deltas.reshape(-1, 1) * 0.5
    
    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    # Equation 13 in NeuS
    alpha = ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).reshape(N_rays_, N_samples_).clip(0.0, 1.0)

    weights = alpha * torch.cumprod(torch.cat([torch.ones([N_rays_, 1],device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]  #cumprod 连乘

    # zyq : add bg_lambda (used in get_results_bg or merge fg bg results)
    if point_type == 'fg':  # get_bg_lambda:  # only when foreground fine =True
        results[f'bg_lambda_{typ}'] = weights[..., -1]

    weights_sum = weights.sum(dim=-1, keepdim=True)
    # calculate color 
    color = color.reshape(N_rays_, N_samples_, 3) # [N, T, 3]
    image = (color * weights[:, :, None]).sum(dim=1)
    if hparams.enable_semantic:
        sem_logits = sem_logits.reshape(N_rays_, N_samples_, sem_logits.shape[-1])
        if hparams.stop_semantic_grad:
            w = weights[..., None].detach()
            sem_map = torch.sum(w * sem_logits, -2)
        else:
            sem_map = torch.sum(weights[..., None] * sem_logits, -2)
        results[f'sem_map_{typ}'] = sem_map

    # calculate normal 
    normal_map = normal.reshape(N_rays_, N_samples_, 3) # [N, T, 3]
    normal_map = torch.sum(normal_map * weights[:, :, None], dim=1)
    
    # # calculate depth 
    ori_z_vals = ((z_vals - near) / (far - near)).clamp(0, 1)
    depth = torch.sum(weights * ori_z_vals, dim=-1)

    # TODO:Eikonal loss 
    pts_norm = torch.linalg.norm(xyz_.reshape(-1, 3), ord=2, dim=-1, keepdim=True).reshape(N_rays_, N_samples_)
    inside_sphere = (pts_norm < 1.0).float().detach()
    relax_inside_sphere = (pts_norm < 1.2).float().detach()

    gradient_error = (torch.linalg.norm(gradient.reshape(N_rays_, N_samples_, 3), ord=2, dim=-1) - 1.0) ** 2
    gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

    assert (gradient == gradient).all(), 'Nan or Inf found!'

    if curvature_loss:
        # TODO:curvature loss 
        random_vec = 2.0 * torch.randn_like(normal) - 1.0
        random_vec_norm = random_vec / (1e-5 + torch.linalg.norm(random_vec, ord=2, dim=-1,  keepdim = True))

        perturbed_pts = xyz_.reshape(-1, 3) + torch.cross(normal, random_vec_norm) * 0.01 * (1.0 - normal_epsilon_ratio) # naively set perturbed points, 
        perturbed_gradient = nerf.gradient(perturbed_pts.reshape(-1, 3), 0.005 * (1.0 - normal_epsilon_ratio)).squeeze()
        perturbed_normal =  perturbed_gradient / (1e-5 + torch.linalg.norm(perturbed_gradient, ord=2, dim=-1,  keepdim = True))

        curvature_error = (torch.sum(normal * perturbed_normal, dim = -1) - 1.0) ** 2
        curvature_error = (relax_inside_sphere * curvature_error.reshape(N_rays_, N_samples_)).sum() / (relax_inside_sphere.sum() + 1e-5)
    else:
        curvature_error = torch.tensor(0.0).cuda()

    # # mix background color
    # if bg_color is None:
    #     bg_color = 1

    # image = image + (1 - weights_sum) * bg_color
    
    # depth = depth.reshape(B, N)
    # image = image.reshape(B, N, 3)
    # depth = None # 暂时不考虑depth， instant-nsr中depth未参与训练

    results['inv_s'] = inv_s_ori
    results[f'rgb_{typ}'] = image
    results[f'depth_{typ}'] = depth
    results[f'normal_map_{typ}'] = normal_map
    results[f'gradient_error_{typ}'] = gradient_error.unsqueeze(0)
    results[f'curvature_error_{typ}'] = curvature_error.unsqueeze(0)
    results['sdf'] = sdf.detach()

    return results #depth, image, normal_map, gradient_error, curvature_error


def _get_results_bg(point_type,
                 nerf: nn.Module,
                 rays_d: torch.Tensor,
                 image_indices: Optional[torch.Tensor],
                 hparams: Namespace,
                 xyz_coarse: torch.Tensor,
                 z_vals: torch.Tensor,
                 last_delta: torch.Tensor,
                 get_depth: bool,
                 get_depth_variance: bool,
                 get_bg_lambda: bool,
                 depth_real: Optional[torch.Tensor],
                 xyz_fine_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]],
                 train_iterations=-1,
                 gt_depths=None,
                 depth_scale=None) -> Dict[str, torch.Tensor]:
    #print('bg')
    results = {}
    #对于bg_nerf来说不变，对fg_nerf来说，属于bg的点的last_delta会变成 交点 - z_vals
    last_delta_diff = torch.zeros_like(last_delta)
    last_delta_diff[last_delta.squeeze() < 1e10, 0] = z_vals[last_delta.squeeze() < 1e10].max(dim=-1)[0]

    """ Step 1: Compute coarse sigma and weights"""

    last_delta_coarse = last_delta - last_delta_diff
    typ='coarse'

    N_rays_ = xyz_coarse.shape[0]
    N_samples_ = xyz_coarse.shape[1]
    xyz_ = xyz_coarse.view(-1, xyz_coarse.shape[-1])
    
    # Perform model inference to get rgb and raw sigma
    B = xyz_.shape[0]
    sigma_chunks_coarse = []
    # (N_rays*N_samples_, embed_dir_channels)
    geo_feat_chunks = []
    if hparams.enable_semantic:
        sem_logits_chunks_coarse = []
        for i in range(0, B, hparams.model_chunk_size):
            xyz_chunk = xyz_[i:i + hparams.model_chunk_size]

            model_chunk, geo_feat_chunk, sem_logits_chunk_coarse = nerf.density(point_type, xyz_chunk,train_iterations=train_iterations)
            sigma_chunks_coarse += [model_chunk]
            geo_feat_chunks += [geo_feat_chunk]
            sem_logits_chunks_coarse += [sem_logits_chunk_coarse]
        
        out = torch.cat(sigma_chunks_coarse, 0)
        out = out.view(N_rays_, N_samples_, out.shape[-1])
        sigmas_coarse = out[..., 0]
        out_geo = torch.cat(geo_feat_chunks, 0)
        geo_feat_coarse = out_geo.view(N_rays_, N_samples_, out_geo.shape[-1])
        sem_logits_coarse = torch.cat(sem_logits_chunks_coarse, 0)
        sem_logits_coarse = sem_logits_coarse.view(N_rays_, N_samples_, sem_logits_coarse.shape[-1])

    else:

        for i in range(0, B, hparams.model_chunk_size):
            xyz_chunk = xyz_[i:i + hparams.model_chunk_size]
            model_chunk, geo_feat_chunk = nerf.density(point_type, xyz_chunk,train_iterations=train_iterations)
            sigma_chunks_coarse += [model_chunk]
            geo_feat_chunks += [geo_feat_chunk]
        
        out = torch.cat(sigma_chunks_coarse, 0)
        out = out.view(N_rays_, N_samples_, out.shape[-1])
        sigmas_coarse = out[..., 0]
        out_geo = torch.cat(geo_feat_chunks, 0)
        geo_feat_coarse = out_geo.view(N_rays_, N_samples_, out_geo.shape[-1])
    
    # Convert these values using volume rendering (Section 4)

    """ gy: the following 7 lines does not grad if fine_samples > 0 """
    deltas_coarse = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
    deltas_coarse = torch.cat([deltas_coarse, last_delta_coarse], -1)  # (N_rays, N_samples_)

    alphas_coarse = 1 - torch.exp(-deltas_coarse * sigmas_coarse)  # (N_rays, N_samples_)
    T_coarse = torch.cumprod(1 - alphas_coarse + 1e-8, -1)
    T_coarse = torch.cat((torch.ones_like(T_coarse[..., 0:1]), T_coarse[..., :-1]), dim=-1)  # [..., N_samples]

    weights_coarse = alphas_coarse * T_coarse  # (N_rays, N_samples_)
    results[f'zvals_{typ}'] = z_vals

    if hparams.fine_samples > 0:  # sample points for fine model

        """" Step 2: Sample fine point """
        typ = 'fine'
        with torch.no_grad():
            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
            perturb = hparams.perturb if nerf.training else 0
            if point_type == 'fg':
                fine_sample = hparams.fine_samples
            elif point_type == 'bg_same_as_fg':
                fine_sample = hparams.fine_samples // 2
            else:
                fine_sample = hparams.fine_samples // 2
            fine_z_vals = _sample_pdf(z_vals_mid, weights_coarse[:, 1:-1].detach(), fine_sample,
                                      det=(perturb == 0))


            xyz_fine, depth_real_fine = xyz_fine_fn(fine_z_vals)
            xyz_fine = contract_to_unisphere(xyz_fine, hparams)
            last_delta_diff = torch.zeros_like(last_delta)
            last_delta_diff[last_delta.squeeze() < 1e10, 0] = fine_z_vals[last_delta.squeeze() < 1e10].max(dim=-1)[0]
            last_delta_fine = last_delta - last_delta_diff
    
        """ Step 3: Compute fine sigma """
        N_rays_ = xyz_fine.shape[0]
        N_samples_ = xyz_fine.shape[1]
        xyz_ = xyz_fine.view(-1, xyz_fine.shape[-1])

        B = xyz_.shape[0]
        out_chunks = []
        geo_feat_chunks = []
        if hparams.enable_semantic:
            sem_logits_chunks_fine = []
            for i in range(0, B, hparams.model_chunk_size):
                xyz_chunk = xyz_[i:i + hparams.model_chunk_size]
                model_chunk, geo_feat_chunk, sem_logits_chunk_fine = nerf.density(point_type, xyz_chunk,train_iterations=train_iterations)
                out_chunks += [model_chunk]
                geo_feat_chunks += [geo_feat_chunk]
                sem_logits_chunks_fine += [sem_logits_chunk_fine]

            out = torch.cat(out_chunks, 0)
            out = out.view(N_rays_, N_samples_, out.shape[-1])
            sigmas_fine = out[..., 0]
            out_geo = torch.cat(geo_feat_chunks, 0)
            geo_feat_fine = out_geo.view(N_rays_, N_samples_, out_geo.shape[-1])
            sem_logits_fine = torch.cat(sem_logits_chunks_fine, 0)
            sem_logits_fine = sem_logits_fine.view(N_rays_, N_samples_, sem_logits_fine.shape[-1])

        else:
            for i in range(0, B, hparams.model_chunk_size):
                xyz_chunk = xyz_[i:i + hparams.model_chunk_size]
                model_chunk, geo_feat_chunk = nerf.density(point_type, xyz_chunk,train_iterations=train_iterations)
                out_chunks += [model_chunk]
                geo_feat_chunks += [geo_feat_chunk]
            out = torch.cat(out_chunks, 0)
            out = out.view(N_rays_, N_samples_, out.shape[-1])
            sigmas_fine = out[..., 0]
            out_geo = torch.cat(geo_feat_chunks, 0)
            geo_feat_fine = out_geo.view(N_rays_, N_samples_, out_geo.shape[-1])

        " Step 4: combine coarse and fine sigma"
        z_vals = torch.cat([z_vals, fine_z_vals], dim=-1)  # [N, T+t]
        # z_vals, z_index = torch.sort(z_vals, dim=1)
        z_vals, z_index = torch.sort(z_vals, -1, descending = False)

        geo_feat = torch.cat([geo_feat_coarse, geo_feat_fine], dim=1)  # [N, T+t, 15]
        geo_feat = torch.gather(geo_feat, dim=1, index=z_index.unsqueeze(-1).expand_as(geo_feat))
        sigmas = torch.cat([sigmas_coarse, sigmas_fine], dim=1)
        sigmas = torch.gather(sigmas, dim=1, index=z_index.expand_as(sigmas))
        deltas_fine = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        deltas_fine = torch.cat([deltas_fine, last_delta_fine], -1)  # (N_rays, N_samples_
        if hparams.enable_semantic:
            sem_logits = torch.cat([sem_logits_coarse, sem_logits_fine], dim=1)
            sem_logits = torch.gather(sem_logits, dim=1, index=z_index.unsqueeze(-1).expand_as(sem_logits))

        alphas_fine = 1 - torch.exp(-deltas_fine * sigmas)
        T_fine = torch.cumprod(1 - alphas_fine + 1e-8, -1)
        if point_type=='fg': #get_bg_lambda:  # only when foreground fine =True
            results[f'bg_lambda_{typ}'] = T_fine[..., -1]
        T_fine = torch.cat((torch.ones_like(T_fine[..., 0:1]), T_fine[..., :-1]), dim=-1)  # [..., N_samples]
        weights = alphas_fine * T_fine  # (N_rays, N_samples_)



        geo_feat_ = geo_feat.view(-1, geo_feat.shape[-1])
        B = geo_feat_.shape[0]
        N_samples_ = fine_sample*2
        rays_d_ = rays_d.repeat(1, N_samples_, 1).view(-1, rays_d.shape[-1])
        image_indices_ = image_indices.repeat(1, N_samples_, 1).view(-1, 1)
        out_chunks = []
        for i in range(0, B, hparams.model_chunk_size):
            geo_feat_chunk = geo_feat_[i:i + hparams.model_chunk_size]
            xyz_chunk = torch.cat([geo_feat_chunk,
                                   rays_d_[i:i + hparams.model_chunk_size],
                                   image_indices_[i:i + hparams.model_chunk_size]], 1)
            model_chunk = nerf.color(point_type, xyz_chunk,train_iterations=train_iterations)
            out_chunks += [model_chunk]
        out = torch.cat(out_chunks, 0)
        out = out.view(N_rays_, N_samples_, out.shape[-1])
        rgbs = out[..., :3]  # (N_rays, N_samples_, 3)

        results[f'rgb_{typ}'] = (weights.unsqueeze(-1) * rgbs).sum(dim=1)
        if hparams.enable_semantic:
            if hparams.stop_semantic_grad:
                w = weights[..., None].detach()
                sem_map = torch.sum(w * sem_logits, -2)
            else:
                sem_map = torch.sum(weights[..., None] * sem_logits, -2)
            results[f'sem_map_{typ}'] = sem_map
        
        with torch.no_grad():
            if get_depth or get_depth_variance:
                if depth_real is not None:
                    depth_map = (weights * depth_real).sum(dim=1)  # n1 n2 -> n1
                else:
                    depth_map = (weights * z_vals).sum(dim=1)  # n1 n2 -> n1

            if get_depth: 
                results[f'depth_{typ}'] = depth_map
            if get_depth_variance:
                depth_map = (weights * z_vals).sum(dim=1)  # n1 n2 -> n1
                results[f'depth_variance_{typ}'] = (weights * (z_vals - depth_map.unsqueeze(1)).square()).sum(
                    axis=-1)


        for key in INTERMEDIATE_KEYS:
            if key in results:
                del results[key]

    return results


def _intersect_sphere(rays_o: torch.Tensor, rays_d: torch.Tensor, sphere_center: torch.Tensor,
                      sphere_radius: torch.Tensor) -> torch.Tensor:
    if sphere_radius is not None:
        rays_o = (rays_o - sphere_center) / sphere_radius
        rays_d = rays_d / sphere_radius

    '''
    rays_o, rays_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(rays_d * rays_o, dim=-1) / torch.sum(rays_d * rays_d, dim=-1)
    p = rays_o + d1.unsqueeze(-1) * rays_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(rays_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        raise Exception(
            'Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!')
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2


def _depth2pts_outside(rays_o: torch.Tensor, rays_d: torch.Tensor, depth: torch.Tensor, sphere_center: torch.Tensor,
                       sphere_radius: torch.Tensor, include_xyz_real: bool, cluster_2d: bool):
    '''
    rays_o, rays_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    rays_o_orig = rays_o
    rays_d_orig = rays_d
    if sphere_radius is not None:
        rays_o = (rays_o - sphere_center) / sphere_radius
        rays_d = rays_d / sphere_radius

    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(rays_d * rays_o, dim=-1) / torch.sum(rays_d * rays_d, dim=-1)
    p_mid = rays_o + d1.unsqueeze(-1) * rays_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_norm = rays_d.norm(dim=-1)
    ray_d_cos = 1. / ray_d_norm
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = rays_o + (d1 + d2).unsqueeze(-1) * rays_d

    rot_axis = torch.cross(rays_o, p_sphere, dim=-1)
    rot_axis = rot_axis / (torch.norm(rot_axis, dim=-1, keepdim=True) + 1e-8)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)

    # now calculate conventional depth
    depth_real = 1. / (depth + 1e-8) * torch.cos(theta) + d1

    if include_xyz_real:
        if cluster_2d:
            pts = torch.cat(
                (rays_o_orig + rays_d_orig * depth_real.unsqueeze(-1), p_sphere_new, depth.unsqueeze(-1)),
                dim=-1)
        else:
            boundary = rays_o_orig + rays_d_orig * (d1 + d2).unsqueeze(-1)
            pts = torch.cat((boundary.repeat(1, p_sphere_new.shape[1], 1), p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    else:
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    return pts, depth_real


def _expand_and_perturb_z_vals(z_vals, samples, perturb, N_rays):
    z_vals = z_vals.expand(N_rays, samples)
    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    return z_vals


def _sample_pdf(bins: torch.Tensor, weights: torch.Tensor, fine_samples: int, det: bool) -> torch.Tensor:
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        fine_samples: the number of samples to draw from the distribution
        det: deterministic or not
    Outputs:
        samples: the sampled samples
    """
    weights = weights + 1e-8  # prevent division by zero (don't do inplace op!)

    pdf = weights / weights.sum(-1).unsqueeze(-1)  # (N_rays, N_samples_)

    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    return _sample_cdf(bins, cdf, fine_samples, det)


def _sample_cdf(bins: torch.Tensor, cdf: torch.Tensor, fine_samples: int, det: bool) -> torch.Tensor:
    N_rays, N_samples_ = cdf.shape

    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive
    if det:
        u = torch.linspace(0, 1, fine_samples, device=bins.device)
        u = u.expand(N_rays, fine_samples)
    else:
        u = torch.rand(N_rays, fine_samples, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1)
    inds_sampled = inds_sampled.view(inds_sampled.shape[0], -1)  # n1 n2 2 -> n1 (n2 2)

    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(cdf_g.shape[0], -1, 2)  # n1 (n2 2) -> n1 n2 2

    bins_g = torch.gather(bins, 1, inds_sampled)
    bins_g = bins_g.view(bins_g.shape[0], -1, 2)  # n1 (n2 2) -> n1 n2 2

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < 1e-8] = 1  # denom equals 0 means a bin has weight 0,
    # in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples





# copy from Instant- NSR
def up_sample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
    """
    Up sampling give a fixed inv_s
    """
    batch_size, n_samples = z_vals.shape

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
    radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
    inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
    sdf = sdf.reshape(batch_size, n_samples)

    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
    
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

    # ----------------------------------------------------------------------------------------------------------
    # Use min value of [ cos, prev_cos ]
    # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
    # robust when meeting situations like below:
    #
    # SDF
    # ^
    # |\          -----x----...
    # | \        /
    # |  x      x
    # |---\----/-------------> 0 level
    # |    \  /
    # |     \/
    # |
    # ----------------------------------------------------------------------------------------------------------
    prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=sdf.device), cos_val[:, :-1]], dim=-1)
    cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
    cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
    cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

    dist = (next_z_vals - prev_z_vals)
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([batch_size, 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

    z_samples = _sample_pdf(z_vals, weights, n_importance, det=True).detach()

    return z_samples

def cat_z_vals(nerf, rays_o, rays_d, z_vals, new_z_vals, sdf, bound=0, last=False):
    batch_size, n_samples = z_vals.shape
    _, n_importance = new_z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
    if bound != 0:
        pts = pts.clamp(-bound, bound)
    z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
    z_vals, index = torch.sort(z_vals, dim=-1)
    if not last:            
        new_sdf = nerf.forward_sdf(pts.reshape(-1, 3))
        new_sdf = new_sdf[...,:1].reshape(batch_size, n_importance)
        sdf = torch.cat([sdf, new_sdf], dim=-1)
        xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
        index = index.reshape(-1)
        sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

    return z_vals, sdf

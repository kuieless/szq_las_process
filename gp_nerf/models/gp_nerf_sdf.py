from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

# zyq : torch-ngp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from gp_nerf.torch_ngp.encoding import get_encoder
from gp_nerf.torch_ngp.activation import trunc_exp
from gp_nerf.models.Plane_module import get_Plane_encoder

import numpy as np

timer = 0

def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.ReLU(out_f)
    )

class NeRF(nn.Module):
    def __init__(self, pos_xyz_dim: int,  # 12   positional embedding 阶数
                 pos_dir_dim: int,  # 4 positional embedding 阶数
                 layers: int,  # 8
                 skip_layers: List[int],  # [4]
                 layer_dim: int,  # 256
                 appearance_dim: int,  # 48
                 affine_appearance: bool,  # affine_appearance : False
                 appearance_count: int,  # appearance_count  : number of images (for rubble is 1678)
                 rgb_dim: int,  # rgb_dim : 3
                 xyz_dim: int,  # xyz_dim : fg = 3, bg =4
                 sigma_activation: nn.Module, hparams):
        super(NeRF, self).__init__()

        #semantic
        self.enable_semantic = hparams.enable_semantic
        self.stop_semantic_grad = hparams.stop_semantic_grad
        self.use_pano_lift = hparams.use_pano_lift
        if self.enable_semantic:
            self.semantic_linear    = nn.Sequential(fc_block(1 + hparams.geo_feat_dim, layer_dim // 2), nn.Linear(layer_dim // 2, hparams.num_semantic_classes))
            self.semantic_linear_bg = nn.Sequential(fc_block(1 + hparams.geo_feat_dim, layer_dim // 2), nn.Linear(layer_dim // 2, hparams.num_semantic_classes))
            for param in self.semantic_linear.parameters():
                param.requires_grad = True
            for param in self.semantic_linear_bg.parameters():
                param.requires_grad = True

        #sdf 
        print("sdf")
        self.include_input = True
        self.geometric_init = True
        self.weight_norm = True
        self.deviation_net = SingleVarianceNetwork(0.3)
        self.activation = nn.Softplus(beta=100)
        
        #nerf mlp
        self.layer_dim = layer_dim
        print("sigma_net_dim: {}".format(self.layer_dim))
        self.appearance_count = appearance_count
        self.appearance_dim = appearance_dim
        self.num_layers = hparams.num_layers
        self.num_layers_color = hparams.num_layers_color
        self.geo_feat_dim = hparams.geo_feat_dim
        #hash
        base_resolution = hparams.base_resolution
        desired_resolution = hparams.desired_resolution
        log2_hashmap_size = hparams.log2_hashmap_size
        num_levels = hparams.num_levels

        self.fg_bound = 1
        self.bg_bound = 1+hparams.contract_bg_len
        self.xyz_dim = xyz_dim

        #plane
        # self.use_scaling = hparams.use_scaling
        # if self.use_scaling:
        #     if 'quad' in hparams.dataset_path or 'sci' in hparams.dataset_path:
        #         self.scaling_factor_ground = (abs(hparams.sphere_center[1:]) + abs(hparams.sphere_radius[1:])) / hparams.aabb_bound
        #         self.scaling_factor_altitude_bottom = 0
        #         self.scaling_factor_altitude_range = (abs(hparams.sphere_center[0]) + abs(hparams.sphere_radius[0])) / hparams.aabb_bound
        #     else:
        #         self.scaling_factor_ground = (abs(hparams.sphere_center[1:]) + abs(hparams.sphere_radius[1:])) / hparams.aabb_bound
        #         self.scaling_factor_altitude_bottom = 0.5 * (hparams.z_range[0]+ hparams.z_range[1])/ hparams.aabb_bound
        #         self.scaling_factor_altitude_range = (hparams.z_range[1]-hparams.z_range[0]) / (2 * hparams.aabb_bound)


        self.embedding_a = nn.Embedding(self.appearance_count, self.appearance_dim)
        
        desired_resolution_fg = desired_resolution
        encoding = "hashgrid"

        print("use two mlp")
        self.encoder, self.in_dim = get_encoder(encoding, base_resolution=base_resolution,
                                                desired_resolution=desired_resolution_fg,
                                                log2_hashmap_size=log2_hashmap_size, num_levels=num_levels)
        self.encoder_bg, _ = get_encoder(encoding, base_resolution=base_resolution,
                                            desired_resolution=desired_resolution,
                                            log2_hashmap_size=19, num_levels=num_levels)

        self.plane_encoder, self.plane_dim = get_Plane_encoder(hparams)
        self.sdf_net, self.color_net, self.encoder_dir = self.get_nerf_mlp()
        self.sigma_net_bg, self.color_net_bg, self.encoder_dir_bg = self.get_nerf_mlp_bg()

        
    # sdf
    def forward_variance(self):
        inv_s = self.deviation_net(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        return inv_s

    def gradient_neus(self, x):

        x.requires_grad_(True)
        y = self.forward_sdf(x)
        y = y[:, :1]
        if not self.training:
            y.requires_grad_(True)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
        return gradients #.unsqueeze(1)

    #instant nsr
    def gradient(self, x, epsilon=0.0005):
        #not allowed auto gradient, using fd instead
        return self.finite_difference_normals_approximator(x, epsilon)

    def finite_difference_normals_approximator(self, x, epsilon = 0.0005):
        # finite difference
        # f(x+h, y, z), f(x, y+h, z), f(x, y, z+h) - f(x-h, y, z), f(x, y-h, z), f(x, y, z-h)
        pos_x = x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)
        dist_dx_pos = self.forward_sdf(pos_x)
        dist_dx_pos = dist_dx_pos[:,:1]
        pos_y = x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)
        dist_dy_pos = self.forward_sdf(pos_y)
        dist_dy_pos = dist_dy_pos[:,:1]
        pos_z = x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)
        dist_dz_pos = self.forward_sdf(pos_z)
        dist_dz_pos = dist_dz_pos[:,:1]

        neg_x = x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)
        dist_dx_neg = self.forward_sdf(neg_x)
        dist_dx_neg = dist_dx_neg[:,:1]
        neg_y = x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)
        dist_dy_neg = self.forward_sdf(neg_y)
        dist_dy_neg = dist_dy_neg[:,:1]
        neg_z = x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)
        dist_dz_neg = self.forward_sdf(neg_z)
        dist_dz_neg = dist_dz_neg[:,:1]

        return torch.cat([0.5*(dist_dx_pos - dist_dx_neg) / epsilon, 0.5*(dist_dy_pos - dist_dy_neg) / epsilon, 0.5*(dist_dz_pos - dist_dz_neg) / epsilon], dim=-1)

    def forward_sdf(self, x: torch.Tensor, sigma_only: bool = False,
        sigma_noise: Optional[torch.Tensor] = None, train_iterations=-1) -> torch.Tensor:
        # sdf
        # x.requires_grad_(True)
        
        position = x[:, :self.xyz_dim]

        h = self.encoder(position, bound=self.fg_bound)

        # if self.use_scaling:
        #     # with torch.no_grad():
        #         px = (position[:, 0:1] - self.scaling_factor_altitude_bottom)/self.scaling_factor_altitude_range
        #         py = position[:, 1:] / self.scaling_factor_ground
        #         position = torch.cat([px, py], dim=-1)
        #     # visualize_points(position.detach().cpu().numpy())
        plane_feat = self.plane_encoder(position, bound=self.fg_bound)
        h = torch.cat([h, plane_feat], dim=-1)

        if self.include_input:
            h = torch.cat([position, h], dim=-1)

        for l in range(self.num_layers):
            h = self.sdf_net[l](h)
            if l != self.num_layers - 1:
                h = self.activation(h)
                #h = F.relu(h, inplace=True)
        
        sdf_output = h

        return sdf_output

    # semantic 
    def forward_semantic(self, h):
        if self.stop_semantic_grad:
            h_stop = h.detach()
            sem_logits = self.semantic_linear(h_stop)
        else:
            sem_logits = self.semantic_linear(h)
        if self.use_pano_lift:
            sem_logits = torch.nn.functional.softmax(sem_logits, dim=-1)
        return sem_logits



    def forward_color(self, x, d, n, geo_feat, image_indices_):
        # dir
        #d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)
        a = self.embedding_a(image_indices_[:,0].long())
        # color x, 
        h = torch.cat([x, d, n, geo_feat, a], dim=-1)
    
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return color


    def get_nerf_mlp(self):
        encoding_dir = "sphere_harmonics"
        geo_feat_dim = self.geo_feat_dim
        sigma_nets = []
        # zyq: 把sigma net改为sdfnet----------------------------
        for l in range(self.num_layers):
            if l == 0:
                in_dim = self.in_dim + 3 if self.include_input else self.in_dim
                #in_dim = self.in_dim
                if True: #self.use_plane:
                    print("Hash and Plane")
                    in_dim = in_dim + self.plane_dim
                else:
                    print("Hash and no Plane")
            else:
                in_dim = self.layer_dim  # 64
            if l == self.num_layers - 1:  # 最后一层
                out_dim = 1 + geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = self.layer_dim
            # sigma_nets.append(nn.Linear(in_dim, out_dim, bias=False))
            sigma_nets.append(nn.Linear(in_dim, out_dim))

            if self.geometric_init:
                    if l == self.num_layers - 1:
                        torch.nn.init.normal_(sigma_nets[l].weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(sigma_nets[l].bias, 0)     

                    elif l==0:
                        if self.include_input:
                            torch.nn.init.constant_(sigma_nets[l].bias, 0.0)
                            torch.nn.init.normal_(sigma_nets[l].weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                            torch.nn.init.constant_(sigma_nets[l].weight[:, 3:], 0.0)
                        else:
                            torch.nn.init.constant_(sigma_nets[l].bias, 0.0)
                            torch.nn.init.normal_(sigma_nets[l].weight[:, :], 0.0, np.sqrt(2) / np.sqrt(out_dim))

                    else:
                        torch.nn.init.constant_(sigma_nets[l].bias, 0.0)
                        torch.nn.init.normal_(sigma_nets[l].weight[:, :], 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.weight_norm:
                sigma_nets[l] = nn.utils.weight_norm(sigma_nets[l])
        sigma_net = nn.ModuleList(sigma_nets)  # 两层全连接
        #---------------------------------------

        encoder_dir, in_dim_dir = get_encoder(encoding_dir)
        color_nets = []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = in_dim_dir + geo_feat_dim + self.appearance_dim + 6 # 6代表输入坐标xyz和法线n
            else:
                in_dim = self.layer_dim

            if l == self.num_layers_color - 1:  # 最后一层
                out_dim = 3  # 3 rgb
            else:
                out_dim = self.layer_dim

            color_nets.append(nn.Linear(in_dim, out_dim, bias=False))
            if self.weight_norm:
                color_nets[l] = nn.utils.weight_norm(color_nets[l])
        color_net = nn.ModuleList(color_nets)  # 3层全连接
        return sigma_net, color_net, encoder_dir

    def get_nerf_mlp_bg(self):
        encoding_dir = "sphere_harmonics"
        geo_feat_dim = self.geo_feat_dim
        sigma_nets = []
        for l in range(self.num_layers):
            if l == 0:
                in_dim = self.in_dim
                if True: #self.use_plane:
                    print("Hash and Plane")
                    in_dim = in_dim
                else:
                    print("Hash and no Plane")
            else:
                in_dim = self.layer_dim  # 64
            if l == self.num_layers - 1:  # 最后一层
                out_dim = 1 + geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = self.layer_dim
            sigma_nets.append(nn.Linear(in_dim, out_dim, bias=False))

        sigma_net = nn.ModuleList(sigma_nets)  # 两层全连接
        encoder_dir, in_dim_dir = get_encoder(encoding_dir)
        color_nets = []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = in_dim_dir + geo_feat_dim + self.appearance_dim
            else:
                in_dim = self.layer_dim

            if l == self.num_layers_color - 1:  # 最后一层
                out_dim = 3  # 3 rgb
            else:
                out_dim = self.layer_dim

            color_nets.append(nn.Linear(in_dim, out_dim, bias=False))

        color_net = nn.ModuleList(color_nets)  # 3层全连接
        return sigma_net, color_net, encoder_dir

    def density(self, point_type, x: torch.Tensor,train_iterations=-1):
        position = x[:, :self.xyz_dim]
        h = self.encoder_bg(position, bound=self.bg_bound)

        for l in range(self.num_layers):
            h = self.sigma_net_bg[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]
        # semantic 
        if self.enable_semantic:
            if self.stop_semantic_grad:
                h_stop = h.detach()
                sem_logits = self.semantic_linear_bg(h_stop)
            else:
                sem_logits = self.semantic_linear_bg(h)
            if self.use_pano_lift:
                sem_logits = torch.nn.functional.softmax(sem_logits, dim=-1)
            return sigma.unsqueeze(1), geo_feat, sem_logits
        else:
            return sigma.unsqueeze(1), geo_feat

    def color(self, point_type, x: torch.Tensor,train_iterations=-1):

        geo_feat = x[:,0:15]
        # color
        d = x[:, -4:-1]
        d = self.encoder_dir_bg(d)
        a = self.embedding_a(x[:, -1].long())

        h = torch.cat([d, geo_feat, a], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net_bg[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        # h = h.to(torch.float32)
        color = torch.sigmoid(h)

        # return color
        return color

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * torch.exp(self.variance * 10.0)


class Embedding(nn.Module):
    def __init__(self, num_freqs: int, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super(Embedding, self).__init__()

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (num_freqs - 1), num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out += [torch.sin(freq * x), torch.cos(freq * x)]

        return torch.cat(out, -1)


class ShiftedSoftplus(nn.Module):
    __constants__ = ['beta', 'threshold']
    beta: int
    threshold: int

    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super(ShiftedSoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x - 1, self.beta, self.threshold)

    def extra_repr(self) -> str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)


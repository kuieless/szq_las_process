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


from tools.sa3d import grid
from nr3d_lib.models.grids.lotd import LoTDEncoding, get_lotd_decoder


def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.ReLU(out_f)
    )

def semantic_mlp(in_f, out_f, dim_mlp, num_hidden):
    semantic_linears = [torch.nn.Linear(in_f, dim_mlp)]
    

    for i in range(num_hidden):
        semantic_linears.append(torch.nn.ReLU(inplace=False))
        # semantic_linears.append(torch.nn.LeakyReLU())
        # semantic_linears.append(torch.nn.PReLU())
        # semantic_linears.append(torch.nn.Softplus())


        semantic_linears.append(torch.nn.Linear(dim_mlp, dim_mlp))
        
    semantic_linears.append(torch.nn.ReLU(inplace=False))
    # semantic_linears.append(torch.nn.LeakyReLU())
    # semantic_linears.append(torch.nn.PReLU())
    # semantic_linears.append(torch.nn.Softplus())

    semantic_linears.append(torch.nn.Linear(dim_mlp, out_f))
    return torch.nn.Sequential(*semantic_linears)


class NeRF(nn.Module):
    def __init__(self, pos_xyz_dim: int,  # 12   positional embedding 
                 pos_dir_dim: int,  # 4 positional embedding 
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
        
        
        #sam
        self.dataset_type = hparams.dataset_type
        print(f"the dataset_type is :{self.dataset_type}")

        self.layer_dim = layer_dim
        print("layer_dim: {}".format(self.layer_dim))
        self.appearance_count = appearance_count
        self.appearance_dim = appearance_dim
        self.num_layers = hparams.num_layers
        self.num_layers_color = hparams.num_layers_color
        self.geo_feat_dim = hparams.geo_feat_dim

        self.semantic_layer_dim = hparams.semantic_layer_dim
        self.separate_semantic = hparams.separate_semantic
        self.semantic_net_type = hparams.semantic_net_type

        
        self.embedding_xyz = Embedding(pos_xyz_dim)
        in_channels_xyz = xyz_dim + xyz_dim * pos_xyz_dim * 2
        self.num_layers_semantic_hidden = hparams.num_layers_semantic_hidden
        print("semantic layer_dim: {}".format(self.semantic_layer_dim))


        #hash
        base_resolution = hparams.base_resolution
        desired_resolution = hparams.desired_resolution
        log2_hashmap_size = hparams.log2_hashmap_size
        num_levels = hparams.num_levels

        self.fg_bound = 1
        self.bg_bound = 1+hparams.contract_bg_len
        self.xyz_dim = xyz_dim
        
       

        self.embedding_a = nn.Embedding(self.appearance_count, self.appearance_dim)
        
        desired_resolution_fg = desired_resolution
        encoding = "hashgrid"
        
        print("use two mlp")

        # self.encoder, self.in_dim = get_encoder(encoding, base_resolution=base_resolution,
        #                                         desired_resolution=desired_resolution_fg,
        #                                         log2_hashmap_size=log2_hashmap_size, num_levels=num_levels)
        
        # # from street-surf: define a cuboid hash-grid
        
        self.dtype = torch.float
        self.device = torch.device("cuda")
        self.n_rgb_used_output = self.geo_feat_dim

        lotd_auto_compute_cfg = {}
        lotd_auto_compute_cfg['type'] = 'ngp'
        lotd_auto_compute_cfg['target_num_params'] = 32 * (2**log2_hashmap_size)
        lotd_auto_compute_cfg['min_res'] = base_resolution
        lotd_auto_compute_cfg['n_feats'] = 2
        lotd_auto_compute_cfg['log2_hashmap_size'] = log2_hashmap_size
        lotd_auto_compute_cfg['max_num_levels'] = None
        param_init_cfg = {}
        param_init_cfg['method'] = 'uniform_to_type'
        param_init_cfg['bound'] = 0.0001
        anneal_cfg = {}
        anneal_cfg['type'] = 'hardmask'
        anneal_cfg['start_it'] = 0
        anneal_cfg['start_level'] = 2
        anneal_cfg['stop_it'] = 4000

        encoding_cfg = {}
        encoding_cfg['lotd_use_cuboid'] = True
        encoding_cfg['lotd_auto_compute_cfg'] = lotd_auto_compute_cfg
        encoding_cfg['param_init_cfg'] = param_init_cfg
        encoding_cfg['anneal_cfg'] = anneal_cfg
        encoding_cfg['aabb'] = hparams.stretch  # 只需要知道长宽高，随后用来分配长方体hash不同轴的分辨率
        encoding_cfg['bounding_size'] = 1.0   # aabb生效时，这个参数不起作用
        # 这里的encoding只包括了dense+hash的特征，不包括原始输入坐标
        self.encoding = LoTDEncoding(3, **encoding_cfg, dtype=self.dtype, device=self.device)
        self.n_rgb_used_extrafeat = self.encoding.out_features
        decoder_cfg = {}
        decoder_cfg['type'] = 'mlp'
        decoder_cfg['D'] = 1
        decoder_cfg['W'] = layer_dim
        decoder_cfg['use_tcnn_backend'] = False
        decoder_activation = {}
        decoder_activation['type'] = 'softplus'
        decoder_activation['beta'] = 100.0
        decoder_cfg['activation'] = decoder_activation

        self.in_dim = self.encoding.out_features
        # plane feature
        self.plane_encoder, self.plane_dim = get_Plane_encoder(hparams)
        self.use_extra_embed = True
        

        self.decoder, self.decoder_type = get_lotd_decoder(self.encoding.lod_meta, (1+self.n_rgb_used_output), 
                                                           n_extra_embed_ch=self.plane_dim ,**decoder_cfg, 
                                                           dtype=self.dtype, device=self.device)
        
        # NOTE: For lotd-annealing, set zero to non-active part of decoder input at start
        if self.encoding.annealer is not None:
            start_level = self.encoding.annealer.start_level
            start_n_feats = sum(self.encoding.lotd.level_n_feats[:start_level+1])
        with torch.no_grad():
            nn.init.zeros_(self.decoder.layers[0].weight[:, start_n_feats:])



        
        self.encoder_bg, self.bg_in_dim = get_encoder(encoding, base_resolution=base_resolution,
                                            desired_resolution=desired_resolution,
                                            log2_hashmap_size=19, num_levels=num_levels)

        self.sigma_net, self.color_net, self.encoder_dir = self.get_nerf_mlp()
        self.sigma_net_bg, self.color_net_bg, self.encoder_dir_bg = self.get_nerf_mlp(nerf_type='bg')



        #semantic
        self.enable_semantic = hparams.enable_semantic
        self.stop_semantic_grad = hparams.stop_semantic_grad
        self.use_pano_lift = hparams.use_pano_lift
        if self.enable_semantic:
            self.use_mask_type = hparams.use_mask_type
            self.num_semantic_classes = hparams.num_semantic_classes
            if self.use_mask_type == 'densegrid':
                self.seg_mask_grid = grid.create_grid(
                'DenseGrid', channels=hparams.num_semantic_classes, world_size=torch.tensor([375,333,261]),
                xyz_min=torch.tensor([-1.4360, -1.2948, -1.000]), xyz_max=torch.tensor([1.4386, 1.2588, 1.0000]))
            
            elif self.use_mask_type == 'densegrid_mlp':
                with torch.enable_grad():
                    self.seg_mask_grid = grid.create_grid(
                    'DenseGrid', channels=hparams.densegird_mlp_dim, world_size=torch.tensor([375,333,261]),
                    xyz_min=torch.tensor([-1.4360, -1.2948, -1.000]), xyz_max=torch.tensor([1.4386, 1.2588, 1.0000]))
                
                self.mask_view_counts = torch.zeros_like(self.seg_mask_grid.grid, requires_grad=False, device='cuda')
                self.mask_linear = torch.nn.Linear(hparams.densegird_mlp_dim, hparams.num_semantic_classes)
                self.mask_linear.weight.requires_grad_(False)
                self.mask_linear.bias.requires_grad_(False)
                # nn.init.zeros_(self.mask_linear.bias)
            elif self.use_mask_type == 'hashgrid_mlp':
                seg_mask_grid, self.seg_mask_grids_dim = get_encoder("hashgrid", base_resolution=64, desired_resolution=1024, log2_hashmap_size=19, num_levels=2, level_dim=1)
                self.seg_mask_grids = torch.nn.ModuleList([seg_mask_grid for i in range(self.num_semantic_classes)])
                self.mask_linears = torch.nn.ModuleList([torch.nn.Linear(self.seg_mask_grids_dim, 1) for i in range(self.num_semantic_classes)])
                for linear in self.mask_linears:
                    linear.weight.requires_grad_(False)
                    linear.bias.requires_grad_(False)
            else:
                if self.separate_semantic:
                    print('separate the semantic mlp from nerf')
                    print(f'semantic_net_type: {self.semantic_net_type}')
                    if self.semantic_net_type == 'mlp':
                        self.semantic_linear = semantic_mlp(in_channels_xyz, hparams.num_semantic_classes, self.semantic_layer_dim, self.num_layers_semantic_hidden)
                        self.semantic_linear_bg = semantic_mlp(in_channels_xyz, hparams.num_semantic_classes, self.semantic_layer_dim, self.num_layers_semantic_hidden)
                    elif self.semantic_net_type == 'hashgrid_mlp':
                        # # 这里semantic分支也用hash+mlp，但mlp的结构和color_net一样
                        self.semantic_hash_encoding = LoTDEncoding(3, **encoding_cfg, dtype=self.dtype, device=self.device)
                        semantic_mlp_in_dim = self.semantic_hash_encoding.out_features
                        self.semantic_linear = semantic_mlp(semantic_mlp_in_dim, hparams.num_semantic_classes, self.layer_dim, self.num_layers_color)
                        
                        self.semantic_hash_encoding_bg = LoTDEncoding(3, **encoding_cfg, dtype=self.dtype, device=self.device)
                        semantic_mlp_in_dim_bg = self.semantic_hash_encoding_bg.out_features
                        self.semantic_linear_bg = semantic_mlp(semantic_mlp_in_dim_bg, hparams.num_semantic_classes, self.layer_dim, self.num_layers_color)
                            
                else:
                    print('add the semantic head to nerf')
                    self.semantic_linear = nn.Sequential(fc_block(1 + self.geo_feat_dim + in_channels_xyz, self.semantic_layer_dim), nn.Linear(self.semantic_layer_dim, hparams.num_semantic_classes))
                    self.semantic_linear_bg = nn.Sequential(fc_block(1 + self.geo_feat_dim + in_channels_xyz, self.semantic_layer_dim), nn.Linear(self.semantic_layer_dim, hparams.num_semantic_classes))


    def get_nerf_mlp(self, nerf_type='fg'):
        encoding_dir = "sphere_harmonics"
        geo_feat_dim = self.geo_feat_dim
        sigma_nets = []
        for l in range(self.num_layers):
            if l == 0:
                print("Hash and Plane")
                if nerf_type == 'fg':
                    in_dim = self.in_dim + self.plane_dim
                else:
                    in_dim = self.bg_in_dim
                
            else:
                in_dim = self.layer_dim  # 64
            if l == self.num_layers - 1:  
                out_dim = 1 + geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = self.layer_dim
            sigma_nets.append(nn.Linear(in_dim, out_dim, bias=False))

        sigma_net = nn.ModuleList(sigma_nets)  
        encoder_dir, in_dim_dir = get_encoder(encoding_dir)
        color_nets = []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = in_dim_dir + geo_feat_dim + self.appearance_dim
                if nerf_type == 'fg':
                    in_dim = in_dim + self.plane_dim
            else:
                in_dim = self.layer_dim

            if l == self.num_layers_color - 1: 
                out_dim = 3  # rgb
            else:
                out_dim = self.layer_dim

            color_nets.append(nn.Linear(in_dim, out_dim, bias=False))

        color_net = nn.ModuleList(color_nets)  
        return sigma_net, color_net, encoder_dir

    def mask_fc_hash(self, logits):
        outs = []
        for i in range(self.num_semantic_classes):
            out = self.mask_linears[i](logits[:,i*self.seg_mask_grids_dim:(i+1)*self.seg_mask_grids_dim])
            outs.append(out)
        outs = torch.cat(outs, dim=-1)
        return outs
    
    # @torch.no_grad()
    def mask_fc_dense(self, x):
        out = self.mask_linear(x)
        return out
    # @torch.no_grad()
    
    def forward(self, point_type, x: torch.Tensor, sigma_only: bool = False,
                sigma_noise: Optional[torch.Tensor] = None,train_iterations=-1) -> torch.Tensor:
        
        if point_type == 'fg':
            out = self.forward_fg(point_type, x, sigma_only, sigma_noise,train_iterations=train_iterations)
        elif point_type == 'bg':
            out = self.forward_bg(point_type, x, sigma_only, sigma_noise,train_iterations=train_iterations)
        else:
            NotImplementedError('Unkonwn point type')
        return out
    
    def forward_fg(self, point_type, x: torch.Tensor, sigma_only: bool = False, sigma_noise: Optional[torch.Tensor] = None,train_iterations=-1) -> torch.Tensor:

        position = x[:, :self.xyz_dim]
        # nr3d
        h = self.encoding(position, max_level=None)
        
        plane_feat = self.plane_encoder(position, bound=self.fg_bound)
        h = torch.cat([h, plane_feat], dim=-1)  #####

        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        sigma = trunc_exp(h[..., 0])   ######
        geo_feat = h[..., 1:]

        if sigma_only:
            return sigma

        # semantic 
        if self.enable_semantic:
            if self.use_mask_type == 'densegrid':
                sem_logits = self.seg_mask_grid(x[:, :self.xyz_dim])
            elif self.use_mask_type == 'densegrid_mlp':
                with torch.enable_grad():
                    sem_logits = self.seg_mask_grid(x[:, :self.xyz_dim])
            elif self.use_mask_type == 'hashgrid_mlp':
                sem_logits = []
                for seg_mask_grid in self.seg_mask_grids:
                    sem_logit = seg_mask_grid(x[:, :self.xyz_dim], bound=1.5)
                    sem_logits.append(sem_logit)
                sem_logits = torch.cat(sem_logits, dim=-1)

            else:
                input_xyz = self.embedding_xyz(x[:, :self.xyz_dim]) 
                if self.separate_semantic:
                    if self.semantic_net_type == 'mlp':
                        sem_feature = self.semantic_linear[:-2](input_xyz)  
                        sem_logits = self.semantic_linear[-2:](sem_feature)   
                    elif self.semantic_net_type == 'hashgrid_mlp':
                        semantic_hash = self.semantic_hash_encoding(x[:, :self.xyz_dim], max_level=None)
                        sem_logits = self.semantic_linear(semantic_hash)  
                    
                else:
                    print('add the semantic head to nerf')
                    if self.stop_semantic_grad:
                        h_stop = h.detach()
                        sem_logits = self.semantic_linear(torch.cat([h_stop, input_xyz], dim=-1))
                    else:
                        sem_logits = self.semantic_linear(torch.cat([h, input_xyz], dim=-1))
                if self.use_pano_lift:
                    sem_logits = torch.nn.functional.softmax(sem_logits, dim=-1)

        # color
        d = x[:, self.xyz_dim:-1]
        d = self.encoder_dir(d)
        a = self.embedding_a(x[:, -1].long())
        h = torch.cat([d, geo_feat, a, plane_feat], dim=-1)  

        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        if self.enable_semantic:
            if self.dataset_type == 'sam':
                return torch.cat([color, sigma.unsqueeze(1)], -1), sem_logits, sem_feature
            else:
                return torch.cat([color, sigma.unsqueeze(1)], -1), sem_logits
        else:
            return torch.cat([color, sigma.unsqueeze(1)], -1)

    def forward_bg(self, point_type, x: torch.Tensor, sigma_only: bool = False, sigma_noise: Optional[torch.Tensor] = None,train_iterations=-1) -> torch.Tensor:
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
            if self.use_mask_type == 'densegrid':
                sem_logits = self.seg_mask_grid(x[:, :self.xyz_dim])
            elif self.use_mask_type == 'densegrid_mlp':
                with torch.enable_grad():
                    sem_logits = self.seg_mask_grid(x[:, :self.xyz_dim])

            elif self.use_mask_type == 'hashgrid_mlp':
                sem_logits = []
                for seg_mask_grid in self.seg_mask_grids:
                    sem_logit = seg_mask_grid(x[:, :self.xyz_dim], bound=1.5)
                    sem_logits.append(sem_logit)
                sem_logits = torch.cat(sem_logits, dim=-1)
            else:
                input_xyz = self.embedding_xyz(x[:, :self.xyz_dim])
                if self.separate_semantic:
                    if self.semantic_net_type == 'mlp':
                        sem_feature = self.semantic_linear_bg[:-2](input_xyz)
                        sem_logits = self.semantic_linear_bg[-2](sem_feature)
                    elif self.semantic_net_type == 'hashgrid_mlp':
                        semantic_hash = self.semantic_hash_encoding_bg(x[:, :self.xyz_dim], max_level=None)
                        sem_logits = self.semantic_linear_bg(semantic_hash) 

                else:
                    if self.stop_semantic_grad:
                        h_stop = h.detach()
                        sem_logits = self.semantic_linear_bg(torch.cat([h_stop, input_xyz], dim=-1))
                    else:
                        sem_logits = self.semantic_linear_bg(torch.cat([h, input_xyz], dim=-1))
                if self.use_pano_lift:
                    sem_logits = torch.nn.functional.softmax(sem_logits, dim=-1)
        


        # color
        d = x[:, self.xyz_dim:-1]
        d = self.encoder_dir_bg(d)
        a = self.embedding_a(x[:, -1].long())
        h = torch.cat([d, geo_feat, a], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net_bg[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        # sigmoid activation for rgb
        color = torch.sigmoid(h)
        if self.enable_semantic:
            if self.dataset_type == 'sam':
                return torch.cat([color, sigma.unsqueeze(1)], -1), sem_logits, sem_feature
            else:
                return torch.cat([color, sigma.unsqueeze(1)], -1), sem_logits
        else:
            return torch.cat([color, sigma.unsqueeze(1)], -1)


    def auto_gradient(self, x, tflag=True):
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            y = self.forward_fg('fg', x, sigma_only=True)
            #y = y[:, :1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=tflag,
                retain_graph=tflag,
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
        dist_dx_pos = self.forward_fg('fg', pos_x, sigma_only=True)
        dist_dx_pos = dist_dx_pos[:, None]
        pos_y = x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)
        dist_dy_pos = self.forward_fg('fg', pos_y, sigma_only=True)
        dist_dy_pos = dist_dy_pos[:,None]
        pos_z = x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)
        dist_dz_pos = self.forward_fg('fg', pos_z, sigma_only=True)
        dist_dz_pos = dist_dz_pos[:,None]

        neg_x = x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)
        dist_dx_neg = self.forward_fg('fg', neg_x, sigma_only=True)
        dist_dx_neg = dist_dx_neg[:,None]
        neg_y = x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)
        dist_dy_neg = self.forward_fg('fg', neg_y, sigma_only=True)
        dist_dy_neg = dist_dy_neg[:,None]
        neg_z = x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)
        dist_dz_neg = self.forward_fg('fg', neg_z, sigma_only=True)
        dist_dz_neg = dist_dz_neg[:,None]


        return torch.cat([0.5*(dist_dx_pos - dist_dx_neg) / epsilon, 0.5*(dist_dy_pos - dist_dy_neg) / epsilon, 0.5*(dist_dz_pos - dist_dz_neg) / epsilon], dim=-1)


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


from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

# zyq : torch-ngp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * torch.exp(self.variance * 10.0)



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


        layer_dim = hparams.layer_dim
        self.xyz_dim = 3
        print(f'pure mlp, layer_dim = {layer_dim}')
        in_channels_xyz = xyz_dim + xyz_dim * pos_xyz_dim * 2
        self.skip_layers = skip_layers
        in_channels_dir = 3 + 3 * pos_dir_dim * 2
        self.embedding_a = nn.Embedding(appearance_count, appearance_dim)
        # output layers
        self.sigma_activation = sigma_activation
        self.rgb_activation = nn.Sigmoid()  # = nn.Sequential(rgb, nn.Sigmoid())

        #sdf 
        self.deviation_net = SingleVarianceNetwork(0.3)
        self.activation = nn.Softplus(beta=100)



        #fg
        self.embedding_xyz = Embedding(pos_xyz_dim)
        self.embedding_dir = Embedding(pos_dir_dim)

        sdf_net = []
        # xyz encoding layers
        for i in range(layers):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, layer_dim)
            elif i in skip_layers:
                layer = nn.Linear(layer_dim + in_channels_xyz, layer_dim)
            else:
                layer = nn.Linear(layer_dim, layer_dim)
            layer = nn.Sequential(layer, nn.ReLU(True))
            sdf_net.append(layer)
        self.sdf_net = nn.ModuleList(sdf_net)
        self.sdf_net_final = nn.Linear(layer_dim, layer_dim+1)
        


        color_net = []
        for i in range(4):
            if i == 0:
                layer = nn.Linear(6 + layer_dim + in_channels_dir + (appearance_dim if not affine_appearance else 0), layer_dim)
            else:
                layer = nn.Linear(layer_dim, layer_dim)
            layer = nn.Sequential(layer, nn.ReLU(True))
            color_net.append(layer)
        self.color_net = nn.ModuleList(color_net)
        self.rgb_final = nn.Linear(layer_dim , 3)




        #bg
        self.embedding_xyz_bg = Embedding(pos_xyz_dim)
        self.sigma_bg = nn.Linear(layer_dim+1, 1)
        self.embedding_dir_bg = Embedding(pos_dir_dim)

        xyz_encodings_bg = []
        # xyz encoding layers
        for i in range(layers):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, layer_dim)
            elif i in skip_layers:
                layer = nn.Linear(layer_dim + in_channels_xyz, layer_dim)
            else:
                layer = nn.Linear(layer_dim, layer_dim)
            layer = nn.Sequential(layer, nn.ReLU(True))
            xyz_encodings_bg.append(layer)
        self.xyz_encodings_bg = nn.ModuleList(xyz_encodings_bg)
        self.xyz_encoding_final_bg = nn.Linear(layer_dim, layer_dim+1)

        # direction and appearance encoding layers
        self.dir_a_encoding_bg = nn.Sequential(
            nn.Linear(layer_dim + in_channels_dir + appearance_dim, layer_dim // 2),
            nn.ReLU(True))

        self.rgb_bg = nn.Linear(layer_dim // 2 , rgb_dim)


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
        
        input_xyz = self.embedding_xyz(x[:, :self.xyz_dim])
        xyz_ = input_xyz
        for i, xyz_encoding in enumerate(self.sdf_net):
            if i in self.skip_layers:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = xyz_encoding(xyz_)
        xyz_ = self.sdf_net_final(xyz_)
        sdf_output = self.activation(xyz_)
        
        return torch.cat([sdf_output[:, :1], sdf_output[:, 1:]], dim=-1)

    def forward_color(self, x, d, n, geo_feat, image_indices_):
        # dir
        #d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.embedding_dir(d)
        a = self.embedding_a(image_indices_[:,0].long())
        # color x, 
        h = torch.cat([x, d, n, geo_feat, a], dim=-1)

        for i, color_net in enumerate(self.color_net):
            h = color_net(h)
        
        h = self.rgb_final(h)
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return color


    def density(self, point_type, x: torch.Tensor,train_iterations=-1):
        input_xyz = self.embedding_xyz_bg(x[:, :self.xyz_dim])
        xyz_ = input_xyz
        for i, xyz_encoding in enumerate(self.xyz_encodings_bg):
            if i in self.skip_layers:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = xyz_encoding(xyz_)
        h = self.xyz_encoding_final_bg(xyz_)
        
        sigma = self.sigma_activation(h[..., 0])
        geo_feat = h[..., 1:]
        

        return sigma.unsqueeze(1), geo_feat

    def color(self, point_type, x: torch.Tensor,train_iterations=-1):
        
        dir_a_encoding_input = [x[:,:256]]
        dir_a_encoding_input.append(self.embedding_dir_bg(x[:, -4:-1]))
        dir_a_encoding_input.append(self.embedding_a(x[:, -1].long()))
        dir_a_encoding = self.dir_a_encoding_bg(torch.cat(dir_a_encoding_input, -1))
        rgb = self.rgb_bg(dir_a_encoding)
        rgb = self.rgb_activation(rgb)

        return rgb
    



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
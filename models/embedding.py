import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import Linear, Conv2d, ReLU
import torch.nn.functional as F
from scipy.signal import convolve2d as convolve
import contextlib
import utils
import copy
import h5py
import numpy as np
import deepxde as dde
from neuralop.models import FNO, FNO1d
from os.path import join
from functools import partial
from models.nithin_embedding import reaction_diff_2d_residual_compute
from models.cn import replace_cn_layers


class ConservedEmbedding(nn.Module):
    def __init__(self, emb_dim=8, image_width=64, nc=3):
        super(ConservedEmbedding, self).__init__()
        self.num_channels = nc
        self.layer1 = nn.Sequential(
            nn.Conv2d(nc, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(4 * image_width * image_width, emb_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out


class ConvConservedEmbedding(nn.Module):
    '''fully convolutional conserved embedding'''

    def __init__(self, image_width=64, nc=3):
        super(ConvConservedEmbedding, self).__init__()
        self.num_channels = nc

        self.layer1 = nn.Sequential(
            nn.Conv2d(nc, 32, kernel_size=5, stride=1, padding=(2, 2)),
            nn.ReLU(),
        )

        # 1D conv
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=(0, 0)),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        return out


class EncoderEmbedding(nn.Module):
    # here for legacy purposes; don't use this
    def __init__(self, encoder, emb_dim=8, image_width=128, nc=3,
                 num_emb_frames=2, batch_size=2):
        super(EncoderEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.num_emb_frames = num_emb_frames
        self.encoder = copy.deepcopy(encoder)
        self.batch_size = batch_size
        self.nc = nc
        replace_cn_layers(self.encoder, batch_size=batch_size * num_emb_frames)
        self.encoder.cuda()
        self.enc_dim = self.encoder.dim
        self.linear = nn.Linear(
            self.enc_dim * self.num_emb_frames, self.emb_dim)

    def forward(self, x):
        out = x.reshape(x.size(0) * self.num_emb_frames,
                        self.nc, x.size(2), x.size(3))
        out = self.encoder(out)[0].reshape(self.batch_size, -1)
        out = self.linear(out)
        return out


class ConstantLayer(nn.Module):
    '''layer that returns a constant value (batched)'''

    def __init__(self, *constants) -> None:
        super().__init__()
        self.num_constants = len(constants)
        self.const_tensor = torch.Tensor(constants).cuda().double()
        self.const_tensor.requires_grad = False

    def forward(self, x):
        b_size = x.shape[0]
        return self.const_tensor.expand(b_size, self.num_constants)


class TwoDDiffusionReactionEmbedding(torch.nn.Module):
    # 2d diffusion-reaction module with learnable parameters
    def __init__(self, in_size, in_channels, n_frames, hidden_channels, n_layers, data_root, learned, num_learned_parameters = 3, use_partials = False):
        super(TwoDDiffusionReactionEmbedding, self).__init__()
        # initialize learnable networks
        self.in_channels = in_channels
        self.in_size = in_size
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.device = torch.cuda.current_device()
        self.num_learned_parameters = num_learned_parameters
        self.use_partials = use_partials
        self.n_frames = n_frames if not self.use_partials else 6*n_frames #(5 partial derivatives + 1 solution field) =  6 

        # param is either a network or a callable constant
        if learned:
            self.paramnet = ParameterNet(
                self.in_size, self.in_channels*self.n_frames, self.hidden_channels, self.n_layers, self.num_learned_parameters).to(self.device)
        else:
            self.paramnet = ConstantLayer(5e-3, 1e-3, 5e-3)
        # initialize grid for finite differences
        try:
            file = h5py.File(join(data_root, "2D_diff-react_Du=0.01704641_Dv=0.0154535_k=0.009386063.h5"))
        except:
            file = h5py.File(join(data_root, '2D_diff-react_Du=0.4978229_Dv=0.3132344_k=0.08693649.h5'))

        self.x = torch.Tensor(file['0001']['grid']['x'][:]).to(self.device)
        self.y = torch.Tensor(file['0001']['grid']['y'][:]).to(self.device)
        self.t = torch.Tensor(file['0001']['grid']['t'][:]).to(self.device)
        self.dx = (self.x[1] - self.x[0])
        self.dy = (self.y[1] - self.y[0])
        self.dt = (self.t[1] - self.t[0])


    def reaction_1(self, solution_field, k):
        u = solution_field[:, -1, 0]
        v = solution_field[:, -1, 1]
        return u - (u * u * u) - k - v

    def reaction_2(self, solution_field):
        u = solution_field[:, -1, 0]
        v = solution_field[:, -1, 1]
        return u - v

    # 2D reaction diffusion
    def forward(self, solution_field, true_params = None, return_params = False):
        # solution_field = solution_field.reshape(solution_field.shape[0],
        #                                         int(solution_field.shape[1] /
        #                                             self.in_channels), self.in_channels,
        #                                         solution_field.shape[2], solution_field.shape[3])
        u_stack = solution_field[:, :, 0]
        v_stack = solution_field[:, :, 1]

        if true_params is not None:
            with torch.no_grad():
                k_true, Du_true, Dv_true = true_params
                true_residual, partials = reaction_diff_2d_residual_compute(u_stack, v_stack, self.x, self.y, self.dt, k_true, Du_true, Dv_true, compute_residual = True, return_partials = True)
        else:
            with torch.no_grad():
                partials = reaction_diff_2d_residual_compute(u_stack, v_stack, self.x, self.y, self.dt, None, None, None, compute_residual = False, return_partials = True)
        #predict params using network
        input_data = torch.cat([solution_field, partials], dim = 1) if self.use_partials else solution_field
        params = self.paramnet(input_data)

        #extract predicted params
        k = params[:, 0]
        #set Du and/or Du to their true values if not learnable
        
        Du = params[:, 1] if self.num_learned_parameters >1 else Du_true
        Dv = params[:, 2] if self.num_learned_parameters >2 else Dv_true
        
        #compute PDE residual
        residual = reaction_diff_2d_residual_compute(u_stack, v_stack, self.x, self.y, self.dt, k, Du, Dv)
        
        if return_params:
            if true_params is not None:
                return residual, true_residual, (k, Du, Dv)
            else:
                return residual, (k, Du, Dv)
        else:
            if true_params is not None:
                return residual, true_residual
            else:
                return residual

        # compute spatial derivatives only on most recent frame (use 4th order central difference scheme)
        
        # last_u = u_stack[:, -1]
        # last_v = v_stack[:, -1]
        # du_xx = (-1*last_u[:, 0:-4, 2:-2] + 16*last_u[:, 1:-3, 2:-2] - 30*last_u[:, 2:-2,
        #          2:-2] + 16*last_u[:, 3:-1, 2:-2] - 1*last_u[:, 4:, 2:-2]) / (12*self.dx**2)
        # du_yy = (-1*last_u[:, 2:-2, 0:-4] + 16*last_u[:, 2:-2, 1:-3] - 30*last_u[:, 2:-2,
        #          2:-2] + 16*last_u[:, 2:-2, 3:-1] - 1*last_u[:, 2:-2, 4:]) / (12*self.dx**2)

        # dv_xx = (-1*last_v[:, 0:-4, 2:-2] + 16*last_v[:, 1:-3, 2:-2] - 30*last_v[:, 2:-2,
        #          2:-2] + 16*last_v[:, 3:-1, 2:-2] - 1*last_v[:, 4:, 2:-2]) / (12*self.dx**2)
        # dv_yy = (-1*last_v[:, 2:-2, 0:-4] + 16*last_v[:, 2:-2, 1:-3] - 30*last_v[:, 2:-2,
        #          2:-2] + 16*last_v[:, 2:-2, 3:-1] - 1*last_v[:, 2:-2, 4:]) / (12*self.dx**2)

        # # pad with zeros
        # du_xx = utils.pad_zeros(du_xx, self.in_size)
        # du_yy = utils.pad_zeros(du_yy, self.in_size)
        # dv_xx = utils.pad_zeros(dv_xx, self.in_size)
        # dv_yy = utils.pad_zeros(dv_yy, self.in_size)

        # # compute time derivatives on stack of frames (use 1st order backward difference scheme)
        # du_t = (last_u - u_stack[:, -2]) / self.dt
        # dv_t = (last_v - v_stack[:, -2]) / self.dt

        # #predict params - all with the same network
        # params = self.paramnet(solution_field)
    
        # k = params[:, 0].unsqueeze(-1).unsqueeze(-1)
        # Du = params[:, 1].unsqueeze(-1).unsqueeze(-1)
        # Dv = params[:, 2].unsqueeze(-1).unsqueeze(-1)

        # # du_dt = Du*du_dxx + Du*du_dyy + Ru
        # eq1 = du_t - self.reaction_1(solution_field,
        #                              k=k) - Du * (du_xx + du_yy)
        # # dv_dt = Dv*dv_dxx + Dv*dv_dyy + Rv
        # eq2 = dv_t - self.reaction_2(solution_field) - \
        #     Dv* (dv_xx + dv_yy)
        # if true_params is not None:
        #     k_true, Du_true, Dv_true = true_params
        #     eq1_true = du_t - self.reaction_1(solution_field,
        #                              k=k_true.unsqueeze(-1).unsqueeze(-1)) - Du_true.unsqueeze(-1).unsqueeze(-1) * (du_xx + du_yy)
            
        #     eq2_true = dv_t - self.reaction_2(solution_field) - \
        #                 Dv_true.unsqueeze(-1).unsqueeze(-1)* (dv_xx + dv_yy)

        # import pdb; pdb.set_trace()
        
        # if return_params:
        #     if true_params is not None:
        #         return (eq1 + eq2)[:, 2:-2, 2:-2], (eq1_true + eq2_true)[:, 2:-2, 2:-2], (k, Du, Dv)
        #     else:
        #         return (eq1 + eq2)[:, 2:-2, 2:-2], (k, Du, Dv)
        # else:
        #     if true_params is not None:
        #         return (eq1 + eq2)[:, 2:-2, 2:-2], (eq1_true + eq2_true)[:, 2:-2, 2:-2]
        #     else:
        #         return (eq1 + eq2)[:, 2:-2, 2:-2]
        


class ParameterNet(nn.Module):
    def __init__(self, in_size, in_channels, hidden_channels, n_layers, num_learned_parameters):
        super(ParameterNet, self).__init__()

        self.fno_encoder = FNO(n_modes=(16, 16), hidden_channels=hidden_channels,
                               in_channels=in_channels, out_channels=hidden_channels, n_layers=n_layers)

        self.conv = nn.Sequential(Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=(2, 2)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(4),
                                  Conv2d(hidden_channels, hidden_channels,
                                         kernel_size=3, stride=1, padding=(2, 2)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(4),
                                  )

        self.mlp = nn.Linear(
            hidden_channels*int(in_size/16)*int(in_size/16), num_learned_parameters)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])

        x = self.fno_encoder(x)
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.mlp(x)

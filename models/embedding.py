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
    def __init__(self, in_size, in_channels, n_frames, hidden_channels, n_layers, data_root, learned):
        super(TwoDDiffusionReactionEmbedding, self).__init__()
        # initialize learnable networks
        self.in_channels = in_channels
        self.in_size = in_size

        # param is either a newtwork or a callable constant
        if learned:
            self.paramnet = ParameterNet(
                in_size, in_channels*n_frames, hidden_channels, n_layers)
        else:
            self.paramnet = ConstantLayer(5e-3, 1e-3, 5e-3)
        # initialize grid for finite differences
        file = h5py.File(join(data_root, "2D_diff-react_NA_NA.h5"))
        x = torch.Tensor(file['0001']['grid']['x'][:])
        y = torch.Tensor(file['0001']['grid']['y'][:])
        t = torch.Tensor(file['0001']['grid']['t'][:])
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dt = t[1] - t[0]

        # initialize 2nd order forward difference approximations
        # self.dxx_op = torch.Tensor([2, -5, 4, -1]).unsqueeze(-1) / (self.dx**3)
        # self.dyy_op = self.dxx_op.T * (self.dx/self.dy)**3
        # self.dt_op = torch.Tensor([-3, 4, -1]).unsqueeze(-1).unsqueeze(-1) / (2*self.dt)

    def reaction_1(self, solution_field, k):
        u = solution_field[:, -1, 0]
        v = solution_field[:, -1, 1]
        return u - (u * u * u) - k.unsqueeze(-1) - v

    def reaction_2(self, solution_field):
        u = solution_field[:, -1, 0]
        v = solution_field[:, -1, 1]
        return u - v

    # 2D reaction diffusion
    def forward(self, solution_field):
        solution_field = solution_field.reshape(solution_field.shape[0],
                                                int(solution_field.shape[1] /
                                                    self.in_channels), self.in_channels,
                                                solution_field.shape[2], solution_field.shape[3])

        # shape: [batch_size, num_emb_frames, num_channels, size, size]                                        )
        u_stack = solution_field[:, :, 0]
        v_stack = solution_field[:, :, 1]

        # compute spatial derivatives only on most recent frame (use 2nd order central difference scheme)
        last_u = u_stack[:, -1]
        last_v = v_stack[:, -1]
        du_xx = (-1*last_u[:, 0:-4, 2:-2] + 16*last_u[:, 1:-3, 2:-2] - 30*last_u[:, 2:-2,
                 2:-2] + 16*last_u[:, 3:-1, 2:-2] - 1*last_u[:, 4:, 2:-2]) / (12*self.dx**2)
        du_yy = (-1*last_u[:, 2:-2, 0:-4] + 16*last_u[:, 2:-2, 1:-3] - 30*last_u[:, 2:-2,
                 2:-2] + 16*last_u[:, 2:-2, 3:-1] - 1*last_u[:, 2:-2, 4:]) / (12*self.dx**2)

        dv_xx = (-1*last_v[:, 0:-4, 2:-2] + 16*last_v[:, 1:-3, 2:-2] - 30*last_v[:, 2:-2,
                 2:-2] + 16*last_v[:, 3:-1, 2:-2] - 1*last_v[:, 4:, 2:-2]) / (12*self.dx**2)
        dv_yy = (-1*last_v[:, 2:-2, 0:-4] + 16*last_v[:, 2:-2, 1:-3] - 30*last_v[:, 2:-2,
                 2:-2] + 16*last_v[:, 2:-2, 3:-1] - 1*last_v[:, 2:-2, 4:]) / (12*self.dx**2)

        # pad with zeros
        du_xx = utils.pad_zeros(du_xx, self.in_size)
        du_yy = utils.pad_zeros(du_yy, self.in_size)
        dv_xx = utils.pad_zeros(dv_xx, self.in_size)
        dv_yy = utils.pad_zeros(dv_yy, self.in_size)

        # compute time derivatives on stack of frames (use 1st order backward difference scheme)
        du_t = (last_u - u_stack[:, -2]) / self.dt
        dv_t = (last_v - v_stack[:, -2]) / self.dt

        params = self.paramnet(solution_field)
        k = params[:, 0]
        Du = params[:, 1]
        Dv = params[:, 2]

        # du_dt = Du*du_dxx + Du*du_dyy + Ru
        eq1 = du_t - self.reaction_1(solution_field,
                                     k=k) - Du.unsqueeze(-1) * (du_xx + du_yy)
        # dv_dt = Dv*dv_dxx + Dv*dv_dyy + Rv
        eq2 = dv_t - self.reaction_2(solution_field) - \
            Dv.unsqueeze(-1) * (dv_xx + dv_yy)

        return (eq1 + eq2)[:, 2:-2, 2:-2]


class ParameterNet(nn.Module):
    def __init__(self, in_size, in_channels, hidden_channels, n_layers):
        super(ParameterNet, self).__init__()

        self.fno_encoder = FNO(n_modes=(16, 16), hidden_channels=hidden_channels,
                               in_channels=in_channels, out_channels=hidden_channels, n_layers=n_layers)

        self.conv = nn.Sequential(Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=(2, 2)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),
                                  Conv2d(hidden_channels, hidden_channels,
                                         kernel_size=3, stride=1, padding=(2, 2)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),
                                  Conv2d(hidden_channels, hidden_channels,
                                         kernel_size=3, stride=1, padding=(2, 2)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2)
                                  )

        self.mlp = nn.Linear(
            hidden_channels*int(in_size/8 + 1)*int(in_size/8 + 1), 3)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])

        x = self.fno_encoder(x)
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.mlp(x)

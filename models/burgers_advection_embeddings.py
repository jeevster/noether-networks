# import sys
# sys.path.append('/home/divyam123/noether_work/noether-networks/') 

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import Linear, Conv2d, ReLU, Conv1d
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
import glob
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


class OneDBurgersEmbedding(torch.nn.Module):
    # 2d diffusion-reaction module with learnable parameters
    def __init__(self, in_size, in_channels, n_frames, hidden_channels, n_layers, data_root, learned, num_learned_parameters = 1, use_partials = False):
        super(OneDBurgersEmbedding, self).__init__()
        # initialize learnable networks
        self.in_channels = in_channels
        self.in_size = in_size
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.device = torch.cuda.current_device()
        self.num_learned_parameters = num_learned_parameters
        self.use_partials = use_partials
        self.n_frames = n_frames if not self.use_partials else 4*n_frames #(3 partial derivatives + 1 solution field) =  4 
        self.data_root = data_root
        # param is either a network or a callable constant
        if learned:
            self.paramnet = ParameterNet(
                self.in_size, self.in_channels*self.n_frames, self.hidden_channels, self.n_layers, self.num_learned_parameters, dimensions=1)
            self.paramnet = self.paramnet.apply(lambda t: t.cuda())
            # self.paramnet = self.paramnet.apply(lambda t: t.double())
        else:
            self.paramnet = ConstantLayer(1e-3)
        # initialize grid for finite differences
        file = h5py.File(join(data_root, glob.glob(f"{self.data_root}/*.hdf5")[-1]))

        self.x = torch.Tensor(file['x-coordinate'][:]).to(self.device)
        self.t = torch.Tensor(file['t-coordinate'][:]).to(self.device)
        self.dx = (self.x[1] - self.x[0])
        self.dt = (self.t[1] - self.t[0])


    def partials_torch(self,data, x, t):
        x_axis = -2
        t_axis = -3

        data_x = torch.gradient(data, spacing = (x,), dim=x_axis)[0]
        data_x_usqr = torch.gradient(data * data / 2, spacing = (x,), dim=x_axis)[0]
        data_xx = torch.gradient(data_x, spacing = (x,), dim=x_axis)[0]
        data_t = torch.gradient(data, spacing = t, dim=t_axis)[0]
        return data_x, data_x_usqr, data_xx, data_t
    
    def burgers_1d_residual_compute(self, u, x,t,nu, return_partials = False):
        nu = nu.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device)
        
        data_x, data_x_usqr, data_xx, data_t = self.partials_torch(u, x, t)
        pi = torch.pi
        # pi = pi.to(torch.float64)
        data_x_usqr = data_x_usqr.to(torch.float64)
        data_x = data_x.to(torch.float64)
        data_t = data_t.to(torch.float64)
        data_xx = data_xx.to(torch.float64)
        eqn1 = data_x_usqr + data_t - ((nu/pi) * data_xx)

        pde_residual = (eqn1.to(torch.float64)).abs().mean(dim = (1,2,3)).to(torch.float64)
        if return_partials:
            u_partials = torch.cat([data_x, data_xx, data_t], dim = 1).to(torch.float64)
            u_partials = u_partials[:,:,None,:,:].to(torch.float64).to(self.device)
            return pde_residual, u_partials #keep u and v partials separate
        else:
            return pde_residual

    def forward(self, solution_field, true_params = None, return_params = False):
        solution_field = solution_field.reshape(solution_field.shape[0],
                                                int(solution_field.shape[1] /
                                                    self.in_channels), self.in_channels,
                                                solution_field.shape[2], 1).to(self.device)

        

        u_stack = solution_field[:, :, 0].to(self.device)
        # u_stack = u_stack.permute((0,1,3,2))
        # print("u_stack", u_stack.shape)

        if true_params is not None:
            with torch.no_grad():
                nu_true = true_params[0]
                true_residual, partials = self.burgers_1d_residual_compute(u_stack,self.x, self.dt, nu_true, return_partials = True)

        input_data = torch.cat([solution_field, partials], dim = 1).to(self.device) if self.use_partials else u_stack
        input_data = input_data.to(torch.float64)
        input_data = input_data.double()
        # pdb.set_trace()
        params = self.paramnet(input_data[:,:,0,:,0])
        nu = params[:, 0].double()
        
        
        # print("NU", nu.shape)
        residual = self.burgers_1d_residual_compute(u_stack,self.x, self.dt, nu)

        
        
        # print("res", residual.shape)
        if return_params:
            if true_params is not None:
                return residual, true_residual, tuple([nu])
            else:
                return residual, tuple([nu])
        else:
            if true_params is not None:
                return residual, true_residual
            else:
                return residual


class ParameterNet(nn.Module):
    def __init__(self, in_size, in_channels, hidden_channels, n_layers, num_learned_parameters, dimensions = 2):
        super(ParameterNet, self).__init__()


        self.dimensions = dimensions
        if self.dimensions == 2:
            self.fno_encoder = FNO(n_modes=(16, 16), hidden_channels=hidden_channels,
            in_channels=in_channels, out_channels=hidden_channels, n_layers=n_layers).double()


            self.conv = nn.Sequential(Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=(2, 2)),
            nn.ReLU().double(),
            nn.MaxPool2d(4).double(),
            Conv2d(hidden_channels, hidden_channels,
            kernel_size=3, stride=1, padding=(2, 2)).double(),
            nn.ReLU().double(),
            nn.MaxPool2d(4).double(),
            ).double()


            self.mlp = nn.Linear(
            hidden_channels*int(in_size/16)*int(in_size/16), num_learned_parameters).double()
        elif self.dimensions == 1:
            self.fno_encoder = FNO1d(n_modes_height = 16, hidden_channels = hidden_channels,
            in_channels = in_channels, out_channels = hidden_channels, n_layers = n_layers).to(torch.complex128)
            self.fno_encoder = self.fno_encoder.apply(lambda t: t.to(torch.complex128))
            self.conv = nn.Sequential(Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool1d(4),
                        Conv1d(hidden_channels, hidden_channels,
                        kernel_size=3, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool1d(4),
                        ).double()

            self.mlp = nn.Linear(
            hidden_channels*int(in_size/16)*int(in_size/16), num_learned_parameters).double()
    def forward(self, x):
        if self.dimensions == 2:
            if len(x.shape) == 5:
                x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])


            x = self.fno_encoder()
            x = self.conv(x).double()
            x = torch.flatten(x, start_dim=1)
            return self.mlp(x)
        elif self.dimensions ==1 :
            # print("inside dim 1")
            x = self.fno_encoder(x)
            x = self.conv(x)
            x = torch.flatten(x, start_dim=1)
            return self.mlp(x)




class OneDAdvectionEmbedding(torch.nn.Module):
    # 2d diffusion-reaction module with learnable parameters
    def __init__(self, in_size, in_channels, n_frames, hidden_channels, n_layers, data_root, learned, num_learned_parameters = 1, use_partials = False):
        super(OneDAdvectionEmbedding, self).__init__()
        # initialize learnable networks
        self.in_channels = in_channels
        self.in_size = in_size
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.device = torch.cuda.current_device()
        self.num_learned_parameters = num_learned_parameters
        self.use_partials = use_partials
        self.n_frames = n_frames if not self.use_partials else 3*n_frames #(2 partial derivatives + 1 solution field) =  3
        self.data_root = data_root
        # param is either a network or a callable constant
        if learned:
            self.paramnet = ParameterNet(
                self.in_size, self.in_channels*self.n_frames, self.hidden_channels, self.n_layers, self.num_learned_parameters, dimensions=1)
            self.paramnet = self.paramnet.apply(lambda t: t.cuda())
            # self.paramnet = self.paramnet.apply(lambda t: t.double())
        else:
            self.paramnet = ConstantLayer(1e-3)
        # initialize grid for finite differences
        file = h5py.File(join(data_root, glob.glob(f"{self.data_root}/*.hdf5")[-1]))

        self.x = torch.Tensor(file['x-coordinate'][:]).to(self.device)
        self.t = torch.Tensor(file['t-coordinate'][:]).to(self.device)
        self.dx = (self.x[1] - self.x[0])
        self.dt = (self.t[1] - self.t[0])


    def partials_torch(self,data, x, t):
        x_axis = -2
        t_axis = -3
        data_x = torch.gradient(data, spacing = (x,), dim=x_axis)[0]
        data_t = torch.gradient(data, spacing = t, dim=t_axis)[0]
        return data_x, data_t
    
    def advection_1d_residual_compute(self, u, x,t,nu, return_partials = False):
        nu = nu.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device)
        
        data_x, data_t = self.partials_torch(u, x, t)
        pi = torch.pi

        eqn1 = (nu * data_x) + data_t 
        " ^^^ advection eqn from pde bench^^^ "

        pde_residual = (eqn1).abs().mean(dim = (1,2,3))

        if return_partials:
            u_partials = torch.cat([data_x, data_t], dim = 1)
            u_partials = u_partials[:,:,None,:,:].to(self.device)
            return pde_residual, u_partials #keep u and v partials separate
        else:
            return pde_residual

    def forward(self, solution_field, true_params = None, return_params = False):
        
        solution_field = solution_field.reshape(solution_field.shape[0],
                                                int(solution_field.shape[1] /
                                                    self.in_channels), self.in_channels,
                                                solution_field.shape[2], 1).to(self.device)

        
        u_stack = solution_field[:, :, 0].to(self.device)
        # u_stack = u_stack.permute((0,1,3,2))

        if true_params is not None:
            with torch.no_grad():
                nu_true = true_params[0]
                true_residual, partials = self.advection_1d_residual_compute(u_stack,self.x, self.dt, nu_true, return_partials = True)

        input_data = torch.cat([solution_field, partials], dim = 1).to(self.device) if self.use_partials else u_stack
        input_data = input_data.to(torch.float64)
        input_data = input_data.double()
        # pdb.set_trace()
        params = self.paramnet(input_data[:,:,0,:,0])
        nu = params[:, 0].double()
        
        
        residual = self.advection_1d_residual_compute(u_stack,self.x, self.dt, nu)

        
        
        if return_params:
            if true_params is not None:
                return residual, true_residual, tuple([nu])
            else:
                return residual, tuple([nu])
        else:
            if true_params is not None:
                return residual, true_residual
            else:
                return residual


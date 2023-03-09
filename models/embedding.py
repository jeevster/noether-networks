import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import Linear, Conv2d, ReLU
import torch.nn.functional as F
import contextlib
import utils
import copy

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

'''fully convolutional conserved embedding'''
class ConvConservedEmbedding(nn.Module):
    def __init__(self, image_width=64, nc=3):
        super(ConvConservedEmbedding, self).__init__()
        self.num_channels = nc

        self.layer1 = nn.Sequential(
            nn.Conv2d(nc, 32, kernel_size=5, stride=1, padding=(2,2)),
            nn.ReLU(),
        )

        #1D conv
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=(0,0)),
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
        self.linear = nn.Linear(self.enc_dim * self.num_emb_frames, self.emb_dim)

    def forward(self, x):
        out = x.reshape(x.size(0) * self.num_emb_frames, self.nc, x.size(2), x.size(3))
        out = self.encoder(out)[0].reshape(self.batch_size, -1)
        out = self.linear(out)
        return out



#2d diffusion-reaction module with learnable parameters
class TwoDDiffusionReactionEmbedding(torch.nn.Module):
    def __init__(self, in_size, in_channels, hidden_channels, n_layers):
        super(TwoDDiffusionReactionEmbedding, self).__init__()

        #initialize learnable networks
        self.k_net = ParameterNet(in_size, in_channels, hidden_channels, n_layers)
        self.d1_net = ParameterNet(in_size, in_channels, hidden_channels, n_layers)
        self.d2_net = ParameterNet(in_size, in_channels, hidden_channels, n_layers)

                                    
    def reaction_1(self, y):
        u1 = y[..., 0].unsqueeze(1)
        u2 = y[..., 1].unsqueeze(1)
        return u1 - (u1 * u1 * u1) - self.k_net(y) - u2

    def reaction_2(self, y):
        u1 = y[..., 0].unsqueeze(1)
        u2 = y[..., 1].unsqueeze(1)
        return u1 - u2

    #2D reaction diffusion
    def forward(self, x, y):
        
        #what are x and y? (I think x is the spatial grid and y is the actual data??)
        #probably should just have x be a fixed thing passed in at the init function
        du1_xx = dde.grad.hessian(y, x, i=0, j=0, component=0)
        du1_yy = dde.grad.hessian(y, x, i=1, j=1, component=0)
        du2_xx = dde.grad.hessian(y, x, i=0, j=0, component=1)
        du2_yy = dde.grad.hessian(y, x, i=1, j=1, component=1)

        # TODO: check indices of jacobian
        du1_t = dde.grad.jacobian(y, x, i=0, j=2)
        du2_t = dde.grad.jacobian(y, x, i=1, j=2)

        eq1 = du1_t - reaction_1(y) - self.d1_net(y) * (du1_xx + du1_yy)
        eq2 = du2_t - reaction_2(y) - self.d2_net(y) * (du2_xx + du2_yy)

        return eq1 + eq2


class ParameterNet(nn.Module):
    def __init__(self, in_size, in_channels, hidden_channels, n_layers):
        super(ParameterNet, self).__init__()

        self.encoder = nn.Sequential(FNO(n_modes=(16, 16), hidden_channels=hidden_channels, \
                                    in_channels=in_channels, out_channels=hidden_channels, n_layers = n_layers),
                                    Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=(0,0)),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=(0,0)),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=(0,0)),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                    )

        self.mlp = nn.Linear(hidden_channels*(in_size/8)*(in_size/8), 1)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim = 1)
        return self.mlp(x)
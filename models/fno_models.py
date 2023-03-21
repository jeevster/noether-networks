import torch.nn as nn

'''FNO Encoder/Decoders - NOT DOING ANYTHING RIGHT NOW'''


class FNOEncoder(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.conv1 = nn.Conv2d(nc, 64, 1)
        self.conv2 = nn.Conv2d(nc, 64, 1)

    def forward(self, x):
        return self.conv1(x)


class FNODecoder(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.conv1 = nn.Conv2d(nc, nc, 1)

    def forward(self, x):
        return self.conv1(x)


'''maps 2x128x128 input to 1x128x128 prior'''


class gaussian_FNO(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(gaussian_FNO, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        # self.embed = nn.Linear(input_size, hidden_size)
        self.FNO = FNO(n_modes=(16, 16), hidden_channels=hidden_size,
                       in_channels=input_size, out_channels=hidden_size, n_layers=n_layers)
        self.mu_net = nn.Linear(hidden_size*hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size*hidden_size, output_size)

    def reparameterize(self, mu, logvar, eps=[None]):
        logvar = logvar.mul(0.5).exp_()

        if eps[0] is None:
            # TODO: check that the param is changed outside of the function
            eps[0] = Variable(logvar.data.new(logvar.size()).normal_())
#             print('resampling eps in reparameterize')

        # NOTE: setting eps to zero for debugging
        # TODO: switch back to deterministic
#         eps[0] = torch.zeros_like(eps[0])
#         print(f'EPS: {eps[0]}')
        return eps[0].mul(logvar).add_(mu)

    def forward(self, input, eps=[None]):
        h_in = self.FNO(input)
        h_in = h_in.view(batch_size, -1)  # flatten
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar, eps=eps)
        return z, mu, logvar

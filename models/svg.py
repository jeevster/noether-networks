from torch import nn
import contextlib
import torch

class SVGModel(nn.Module):
    def __init__(self, encoder, frame_predictor, decoder, prior, posterior, emb=None):
        super().__init__()
        self.encoder = encoder
        self.frame_predictor = frame_predictor
        self.decoder = decoder
        self.prior = prior
        self.posterior = posterior
        self.emb = emb


    def forward(self, x_in, gt=None, skip=None, opt=None, i=None, mode='eval', prior_eps=[None], posterior_eps=[None]):
        '''
        Perform a forward pass of either the SVG model or the embedding layer
        
        Because `higher` is annoying, we use different modes in this forward method
        to perform a forward pass through the embedding (haven't found a way to have
        a call to an fmodel's submodule forward method track higher order grads)
        '''
        if mode == 'eval':
#           print(f'prior_eps beginning of SVGModel forward (eval):  {prior_eps}')
            #h, skip_t = self.encoder(x_in)
            h = self.encoder(x_in) #identity mapping
            # if opt.last_frame_skip or i < opt.n_past:
            #     skip[0] = skip_t

            #get prior
            # z_t, mu_p, logvar_p = self.prior(h, eps=prior_eps)
            # if i < opt.n_eval: #n_eval = n_past + n_future
            #     h_target = self.encoder(gt)[0]
            #     z_t_post, mu, logvar = self.posterior(h_target, eps=posterior_eps)
            #     if i < opt.n_past:
            #         x_hat = gt
            #         z_t = z_t_post
            #else:
            mu, logvar, mu_p, logvar_p, skip = torch.Tensor([0.]), torch.Tensor([0.]), torch.Tensor([0.]), torch.Tensor([0.]), torch.Tensor([0.])

            #predict
            if i >= opt.n_past:
                x_hat = self.frame_predictor(h) #predict
                x_hat = self.decoder(x_hat) #identity mapping
            else: #just predict the gt if we're not in the future yet
                x_hat = gt

            
#            print(f'prior_eps end of SVGModel forward (eval):  {prior_eps}')
            return x_hat, mu, logvar, mu_p, logvar_p, skip
    
        elif mode == 'train':
#             print(f'prior_eps beginning of SVGModel forward (train):  {prior_eps}')
            #h, skip_t = self.encoder(x_in)
            h = self.encoder(x_in) #identity mapping
            # h_target = self.encoder(gt)[0]
            # if opt.last_frame_skip or i < opt.n_past:
            #     skip[0] = skip_t
            # z_t, mu, logvar = self.posterior(h_target, eps=posterior_eps)
            # _, mu_p, logvar_p = self.prior(h, eps=prior_eps)

            mu, logvar, mu_p, logvar_p, skip = torch.Tensor([0.]), torch.Tensor([0.]), torch.Tensor([0.]), torch.Tensor([0.]), torch.Tensor([0.])
            h_pred = self.frame_predictor(h) #predidt
            x_hat = self.decoder(h_pred) #identity mapping

            return x_hat, mu, logvar, mu_p, logvar_p, skip

        elif mode == 'emb':
            return self.emb(x_in)
        raise NotImplementedError('please use either "svg" or "emb" mode')

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

import imageio
from torch.autograd import Variable
from torchvision import datasets, transforms
from models.svg import SVGModel
from models.cn import replace_cn_layers, CNLayer
from torch import nn
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy import signal
from torchmetrics.functional import peak_signal_noise_ratio as psnr_metric
from torchmetrics.functional import structural_similarity_index_measure as ssim_metric
import functools
import matplotlib.pyplot as plt
import math
import torch
import socket
import argparse
import os
import numpy as np
from sklearn.manifold import TSNE
import scipy.misc
import matplotlib
import yaml
import pdb
matplotlib.use('agg')

try:
    from models.embedding import ConservedEmbedding
except:
    pass


hostname = socket.gethostname()


svg_mse_crit = nn.MSELoss()

def dump_params_to_yml(params, filepath):
    with open(os.path.join(filepath, "config.yaml"), 'w') as f:
        yaml.dump(params, f)

def pad_zeros(tensor, target_size):
    # Get the current size of the tensor
    _, height, width = tensor.size()

    # Compute the amount of padding needed along each axis
    height_pad = (target_size - height) // 2
    width_pad = (target_size - width) // 2

    # Pad the tensor with zeros along each axis
    padded = nn.functional.pad(
        tensor, (width_pad, width_pad, height_pad, height_pad), mode='constant', value=0)

    return padded


def svg_kl_crit(mu1, logvar1, mu2, logvar2, opt):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2))
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) +
                                      (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / opt.batch_size

#pde task loss
def svg_pde_crit(gen_seq, true_params, pde_crit, opt):
    embs = []
    param_loss = []
    if opt.num_emb_frames == 1:
        embs = [pde_crit(frame, true_params = true_params)[0] for frame in gen_seq]
    elif opt.num_emb_frames > 1:
        assert(len(gen_seq) >= opt.num_emb_frames)
        pde_value, true_pde_value, pred_params = pde_crit(torch.stack(gen_seq, dim=1), true_params = true_params, return_params = True)
        embs.append(pde_value)
    else:
        raise ValueError
    if opt.outer_loss_choice == 'old' or opt.outer_loss_choice == None or opt.outer_loss_choice == False:
        minimizing_loss = [emb.mean() for emb in embs]
    elif opt.outer_loss_choice == 'abs':
        minimizing_loss = [torch.abs(emb).mean() for emb in embs]
    elif opt.outer_loss_choice == 'log':
        minimizing_loss = [torch.abs(emb).log10().mean() for emb in embs]
    elif opt.outer_loss_choice == 'mse':
        minimizing_loss = [(emb ** 2).mean() for emb in embs]
    elif opt.outer_loss_choice == 'l2norm':
        minimizing_loss = [torch.norm(emb.view(emb.shape[0], -1),2,1).mean() for emb in embs]
    return minimizing_loss, [torch.abs(emb).mean() for emb in embs], [torch.abs(emb).log10().mean() for emb in embs], param_loss
def svg_crit(gen_seq, gt_seq, mus, logvars, mu_ps, logvar_ps, pde_crit, params, opt, plot = False, save_dir = ''):
    # svg_mse_losses = list(svg_mse_crit(gen, gt) for gen, gt in zip(gen_seq[opt.n_past:], gt_seq[opt.n_past:]))
    # 
    # MSE (data) and PDE (PINN) Loss terms for Task loss
    svg_pde_losses, avg_pde_residual, log_pde_residual, param_loss = svg_pde_crit(gen_seq, params, pde_crit, opt)
    svg_mse_losses = list(svg_mse_crit(gen, gt)
                          for gen, gt in zip(gen_seq[1:], gt_seq[1:]))
    data_loss = torch.stack(svg_mse_losses).sum()

    relative_data_losses = [torch.mean(torch.norm(gen.reshape(gen.shape[0], -1) - gt.reshape(gen.shape[0], -1), 2, 1) / torch.norm(gt.reshape(gen.shape[0], -1), 2, 1)) for gen, gt in zip(gen_seq[1:], gt_seq[1:])]
    relative_data_loss = torch.stack(relative_data_losses).mean()

    if plot:
        plt.figure()
        plt.plot(gen_seq[-1][0,0,:,0].detach().cpu().numpy(), label = 'predicted trajectory')
        plt.plot(gt_seq[-1][0,0,:,0].detach().cpu().numpy(), label = 'ground truth')
        plt.title('trajectories comparision')
        plt.legend()
        plt.show()
        plt.savefig(f'{save_dir}/trajectory.jpg')

        f = open(f"{save_dir}/trajectory_residuals.txt", "w")
        integer = avg_pde_residual[-1]
        f.write(str(integer))
        f.close()
        
    if opt.relative_data_loss == True:
        final_opt_loss = relative_data_loss

    else:
        final_opt_loss = data_loss
    final_pde_loss = torch.Tensor([0.])
    if opt.pinn_outer_loss: #include both data and pde loss   [0]
        final_pde_loss = opt.pinn_outer_loss_weight*torch.stack(svg_pde_losses).sum()
    
    # TODO: investigate effect of KL term on tailoring
    # svg_kl_losses = [svg_kl_crit(mu, logvar, mu_p, logvar_p, opt) for mu, logvar, mu_p, logvar_p
    #                  in zip(mus, logvars, mu_ps, logvar_ps)]
    # svg_loss += opt.svg_loss_kl_weight * torch.stack(svg_kl_losses).sum()
    return final_opt_loss, final_pde_loss, torch.stack(avg_pde_residual).sum(), data_loss, relative_data_loss, torch.stack(log_pde_residual).sum()


def load_dataset(opt):
    if opt.dataset == 'phys101':
        from data.phys101 import Phys101
        length = opt.train_set_length + opt.test_set_length
        center_crop = 896
        crop_upper_right = 640
        only_twenty_degree = False
        frame_step = 1
        subseq = 'middle'  # start or middle
        horiz_flip = False
        if hasattr(opt, 'center_crop'):
            center_crop = opt.center_crop
        if hasattr(opt, 'crop_upper_right'):
            crop_upper_right = opt.crop_upper_right
        if hasattr(opt, 'only_twenty_degree') and opt.only_twenty_degree:
            only_twenty_degree = True
        if hasattr(opt, 'frame_step'):
            frame_step = opt.frame_step
        if hasattr(opt, 'frame_step'):
            frame_step = opt.frame_step
        if hasattr(opt, 'subseq'):
            subseq = opt.subseq
        if hasattr(opt, 'horiz_flip'):
            horiz_flip = opt.horiz_flip
        train_data = Phys101(
            data_root=opt.data_root,
            train=True,
            image_size=opt.image_width,
            seq_len=opt.n_eval,
            percent_train=(opt.train_set_length/length),
            center_crop=center_crop,
            crop_upper_right=crop_upper_right,
            only_twenty_degree=only_twenty_degree,
            frame_step=frame_step,
            subseq=subseq,
            length=length,
            horiz_flip=horiz_flip,
        )
        test_data = Phys101(
            data_root=opt.data_root,
            train=False,
            image_size=opt.image_width,
            seq_len=opt.n_eval,
            percent_train=(opt.train_set_length/length),
            center_crop=center_crop,
            crop_upper_right=crop_upper_right,
            only_twenty_degree=only_twenty_degree,
            frame_step=frame_step,
            subseq=subseq,
            length=length,
            horiz_flip=horiz_flip,
        )

    elif opt.dataset == '2d_reacdiff':
        from data.twod_reacdiff import TwoDReacDiff
        length = opt.train_set_length + opt.test_set_length
        frame_step = 1
        if hasattr(opt, 'frame_step'):
            frame_step = opt.frame_step

        train_data = TwoDReacDiff(
            data_root=opt.data_root,
            train=True,
            image_size=opt.image_width,
            seq_len=opt.n_eval,
            percent_train=(opt.train_set_length/length),
            frame_step=frame_step,
            length=length,
        )
        test_data = TwoDReacDiff(
            data_root=opt.data_root,
            train=False,
            image_size=opt.image_width,
            seq_len=opt.n_eval,
            percent_train=(opt.train_set_length/length),
            frame_step=frame_step,
            length=length,
        )

    elif opt.dataset == '2d_reacdiff_multiparam':
        from data.twod_reacdiff import TwoDReacDiff_MultiParam
        frame_step = 1
        if hasattr(opt, 'frame_step'):
            frame_step = opt.frame_step
        train_data = TwoDReacDiff_MultiParam(
            data_root=opt.data_root,
            train=True,
            shuffle=True,
            image_size=opt.image_width,
            seq_len=opt.n_eval if hasattr(opt, 'train_noether') else opt.num_emb_frames,
            percent_train=0.8,
            frame_step=frame_step,
            num_param_combinations=opt.num_param_combinations,
            fixed_ic = opt.fixed_ic,
            fixed_window = opt.fixed_window
        )
        test_data = TwoDReacDiff_MultiParam(
            data_root=opt.data_root,
            train=False,
            shuffle=True,
            image_size=opt.image_width,
            seq_len=opt.n_eval if hasattr(opt, 'train_noether') else opt.num_emb_frames,
            percent_train=0.8,
            frame_step=frame_step,
            num_param_combinations=opt.num_param_combinations,
            fixed_ic = opt.fixed_ic,
            fixed_window = opt.fixed_window
        )
    elif opt.dataset == '1d_burgers_multiparam':
        from data.oned_burger_advection import OneD_Advection_Burgers_MultiParam
        frame_step = 1
        if hasattr(opt, 'frame_step'):
            frame_step = opt.frame_step
        train_data = OneD_Advection_Burgers_MultiParam(
            data_root=opt.data_root,
            train=True,
            pde = 'burgers',
            shuffle=True,
            image_size=opt.image_width,
            seq_len=opt.n_eval if hasattr(opt, 'train_noether') else opt.num_emb_frames,
            percent_train=0.8,
            frame_step=frame_step,
            num_param_combinations=opt.num_param_combinations,
            fixed_ic = opt.fixed_ic,
            fixed_window = opt.fixed_window,
            ood = opt.ood if hasattr(opt, 'ood') else False,
            total_size_percent = opt.total_size_percent if hasattr(opt, 'total_size_percent') else 1,
            percent_trajectories = opt.percent_trajectories if hasattr(opt,'percent_trajectories') else 1
        )
        test_data = OneD_Advection_Burgers_MultiParam(
            data_root=opt.data_root,
            train=False,
            pde = 'burgers',
            shuffle=True,
            image_size=opt.image_width,
            seq_len=opt.n_eval if hasattr(opt, 'train_noether') else opt.num_emb_frames,
            percent_train=0.8,
            frame_step=frame_step,
            num_param_combinations=opt.num_param_combinations,
            fixed_ic = opt.fixed_ic,
            fixed_window = opt.fixed_window,
            ood = opt.ood if hasattr(opt, 'ood') else False,
            total_size_percent = opt.total_size_percent if hasattr(opt, 'total_size_percent') else 1,
            percent_trajectories = opt.percent_trajectories if hasattr(opt,'percent_trajectories') else 1
        )
    elif opt.dataset == '1d_advection_multiparam':
        from data.oned_burger_advection import OneD_Advection_Burgers_MultiParam
        frame_step = 1
        if hasattr(opt, 'frame_step'):
            frame_step = opt.frame_step
        train_data = OneD_Advection_Burgers_MultiParam(
            data_root=opt.data_root,
            train=True,
            pde = 'advection',
            image_size=opt.image_width,
            seq_len=opt.n_eval if hasattr(opt, 'train_noether') else opt.num_emb_frames,
            percent_train=0.8,
            frame_step=frame_step,
            num_param_combinations=opt.num_param_combinations,
            fixed_ic = opt.fixed_ic,
            fixed_window = opt.fixed_window,
            shuffle=True,
            ood = opt.ood if hasattr(opt, 'ood') else False,
            total_size_percent = opt.total_size_percent if hasattr(opt, 'total_size_percent') else 1,
            percent_trajectories = opt.percent_trajectories if hasattr(opt,'percent_trajectories') else 1
        )
        test_data = OneD_Advection_Burgers_MultiParam(
            data_root=opt.data_root,
            train=False,
            pde = 'advection',
            image_size=opt.image_width,
            seq_len=opt.n_eval if hasattr(opt, 'train_noether') else opt.num_emb_frames,
            percent_train=0.8,
            frame_step=frame_step,
            num_param_combinations=opt.num_param_combinations,
            fixed_ic = opt.fixed_ic,
            fixed_window = opt.fixed_window,
            shuffle=True,
            ood = opt.ood if hasattr(opt, 'ood') else False,
            total_size_percent = opt.total_size_percent if hasattr(opt, 'total_size_percent') else 1,
            percent_trajectories = opt.percent_trajectories if hasattr(opt,'percent_trajectories') else 1
        )
    elif opt.dataset == '1d_diffusion_reaction_multiparam':
        
        from data.oned_burger_advection import OneD_Advection_Burgers_MultiParam
        frame_step = 1
        if hasattr(opt, 'frame_step'):
            frame_step = opt.frame_step
        train_data = OneD_Advection_Burgers_MultiParam(
            data_root=opt.data_root,
            train=True,
            pde = 'diffuion_reaction',
            image_size=opt.image_width,
            seq_len=opt.n_eval if hasattr(opt, 'train_noether') else opt.num_emb_frames,
            percent_train=0.8,
            frame_step=frame_step,
            num_param_combinations=opt.num_param_combinations,
            fixed_ic = opt.fixed_ic,
            fixed_window = opt.fixed_window,
            shuffle=True,
            ood = opt.ood if hasattr(opt, 'ood') else False,
            total_size_percent = opt.total_size_percent if hasattr(opt, 'total_size_percent') else 1,
            percent_trajectories = opt.percent_trajectories if hasattr(opt,'percent_trajectories') else 1
        )
        test_data = OneD_Advection_Burgers_MultiParam(
            data_root=opt.data_root,
            train=False,
            pde = 'diffuion_reaction',
            image_size=opt.image_width,
            seq_len=opt.n_eval if hasattr(opt, 'train_noether') else opt.num_emb_frames,
            percent_train=0.8,
            frame_step=frame_step,
            num_param_combinations=opt.num_param_combinations,
            fixed_ic = opt.fixed_ic,
            fixed_window = opt.fixed_window,
            shuffle=True,
            ood = opt.ood if hasattr(opt, 'ood') else False,
            total_size_percent = opt.total_size_percent if hasattr(opt, 'total_size_percent') else 1,
            percent_trajectories = opt.percent_trajectories if hasattr(opt,'percent_trajectories') else 1
        )
    elif opt.dataset == '1d_advection_multiparam_fno_rnn':
        from data.oned_burger_advection import OneD_Advection_Burgers_MultiParam
        frame_step = 1
        if hasattr(opt, 'frame_step'):
            frame_step = opt.frame_step
        train_data = OneD_Advection_Burgers_MultiParam(
            data_root=opt.data_root,
            train=True,
            pde = 'advection',
            image_size=opt.image_width,
            seq_len=opt.n_eval if hasattr(opt, 'train_noether') else opt.num_emb_frames,
            percent_train=0.8,
            frame_step=frame_step,
            num_param_combinations=opt.num_param_combinations,
            fixed_ic = opt.fixed_ic,
            fixed_window = opt.fixed_window,
            shuffle=True,
            fno_rnn = True
        )
        test_data = OneD_Advection_Burgers_MultiParam(
            data_root=opt.data_root,
            train=False,
            pde = 'advection',
            image_size=opt.image_width,
            seq_len=opt.n_eval if hasattr(opt, 'train_noether') else opt.num_emb_frames,
            percent_train=0.8,
            frame_step=frame_step,
            num_param_combinations=opt.num_param_combinations,
            fixed_ic = opt.fixed_ic,
            fixed_window = opt.fixed_window,
            shuffle=True,
            fno_rnn = True
        )
    else:
        raise ValueError(
            'Only "phys101" and "2d_reacdiff" are supported. Other datasets are available in the original SVG repo.'
        )
    return train_data, test_data


def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]


def sequence_stack_input(seq, dtype):
    return [torch.cat((x.type(dtype), y.type(dtype)), 1) for x, y in zip(seq[:-1:2], seq[1::2])]


def normalize_data(opt, dtype, sequence):
    if opt.dataset == 'smmnist' or opt.dataset == 'kth' or \
       opt.dataset == 'bair' or opt.dataset == 'omnipush' or \
       opt.dataset == 'phys101' or opt.dataset == '2d_reacdiff':
        sequence.transpose_(0, 1)
        sequence.transpose_(3, 4).transpose_(2, 3)
    else:
        sequence.transpose_(0, 1)

    if hasattr(opt, 'stack_frames') and opt.stack_frames:
        return sequence_stack_input(sequence, dtype)
    return sequence_input(sequence, dtype)


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))


def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding:
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding:
                   (i+1) * y_dim + i * padding].copy_(image)
        return result


def save_np_img(fname, x):
    if x.shape[0] == 1:
        x = np.tile(x, (3, 1, 1))
    img = scipy.misc.toimage(x,
                             high=255*x.max(),
                             channel_axis=0)
    img.save(fname)


def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    # import pdb
    # 
    return scipy.misc.toimage(tensor.detach().numpy(),
                              high=255*tensor.detach().numpy().max(),
                              channel_axis=0)


def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0, 0, 0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)


def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0, 1).transpose(1, 2).clamp(0, 1)
        images.append(img.numpy())
    imageio.mimsave(filename, images, duration=duration)


def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti)
                           for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0, 1).transpose(1, 2).clamp(0, 1).numpy()
        images.append(img)
    imageio.mimsave(filename, images, duration=duration)


def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)


def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)


def prod(l):
    return functools.reduce(lambda x, y: x * y, l)


def batch_flatten(x):
    return x.resize(x.size(0), prod(x.size()[1:]))


def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")


def mse_metric(x1, x2):
    err = ((x1 - x2) ** 2).sum()
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err


def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = torch.zeros((bs, T)).cuda()
    psnr = torch.zeros((bs, T)).cuda()
    mse = torch.zeros((bs, T)).cuda()
    for i in range(bs):
        for t in range(T):
            for c in range(gt[t][i].shape[0]):
                ssim[i, t] += ssim_metric(gt[t][i][c], pred[t][i][c])
                psnr[i, t] += psnr_metric(gt[t][i][c], pred[t][i][c])
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]
            mse[i, t] = mse_metric(gt[t][i], pred[t][i])

    return mse.cpu().numpy(), ssim.cpu().numpy(), psnr.cpu().numpy()

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.


def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[t][i].shape[0]):
                res = finn_ssim(gt[t][i][c], pred[t][i][c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(gt[t][i][c], pred[t][i][c])
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]
            mse[i, t] = mse_metric(gt[t][i], pred[t][i])

    return mse, ssim, psnr


def finn_psnr(x, y):
    mse = ((x - y)**2).mean()
    return 10*np.log(1/mse)/np.log(10)


def gaussian2(size, sigma):
    A = 1/(2.0*np.pi*sigma**2)
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = A*np.exp(-((x**2/(2.0*sigma**2))+(y**2/(2.0*sigma**2))))
    return g


def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def finn_ssim(img1, img2, cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1  # bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                                                     (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                                                    (sigma1_sq + sigma2_sq + C2))


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # print(f'init_weights: m={m}')
        m.weight.data.normal_(0.0, 0.02)
#         m.weight.data.fill_(0.)
        m.bias.data.fill_(0)
        # print(f'classname: {classname};    {list(m.named_parameters())}')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
#         m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)


def init_forget_bias_to_one(model):
    for name, p in model.named_parameters():
        if 'bias_ih' in name:
            n = p.size(0)
            forget_start_idx, forget_end_idx = n // 4, n // 2
            p[forget_start_idx:forget_end_idx].data.fill_(1)


def save_gif_IROS_2019(gif_fname, images, fps=12):
    """
    To generate a gif from image files, first generate palette from images
    and then generate the gif from the images and the palette.
    ffmpeg -i input_%02d.jpg -vf palettegen -y palette.png
    ffmpeg -i input_%02d.jpg -i palette.png -lavfi paletteuse -y output.gif
    Alternatively, use a filter to map the input images to both the palette
    and gif commands, while also passing the palette to the gif command.
    ffmpeg -i input_%02d.jpg -filter_complex "[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse" -y output.gif
    To directly pass in numpy images, use rawvideo format and `-i -` option.
    """
    from subprocess import Popen, PIPE
    head, tail = os.path.split(gif_fname)
    if head and not os.path.exists(head):
        os.makedirs(head)
    h, w, c = images[0].shape
    cmd = ['ffmpeg', '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-r', '%.02f' % fps,
           '-s', '%dx%d' % (w, h),
           '-pix_fmt', {1: 'gray', 3: 'rgb24', 4: 'rgba'}[c],
           '-i', '-',
           '-filter_complex', '[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse',
           '-r', '%.02f' % fps,
           '%s' % gif_fname]
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in images:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        err = '\n'.join([' '.join(cmd), err.decode('utf8')])
        raise IOError(err)
    del proc


def plot_batch(batch, *args, num_imgs=4):
    if len(args):
        batch = [batch] + list(args)
    plt.figure(figsize=(50, 5*(10+len(args))))

    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    if False:
        plt.imshow(
            np.vstack(
                [np.hstack([gray[i].numpy().transpose((1, 2, 0)).reshape(-1, 64, 1) for i in range(num_imgs)]),
                 np.hstack([graymean[i].numpy().transpose((1, 2, 0)).reshape(-1, 64, 1) for i in range(num_imgs)])]
            ),
            cmap='gray'
        )
    if type(batch) in (list, tuple):
        plt.imshow(
            np.vstack(
                [
                    np.hstack([b.cpu()[i].numpy().transpose((1, 2, 0)) for i in range(num_imgs)]) for b in batch
                ]
            )
        )
    else:
        plt.imshow(
            np.hstack([batch.cpu()[i].numpy().transpose((1, 2, 0))
                      for i in range(num_imgs)])
        )


def load_base_weights(model, base_state_dict):
    '''
    Loads weights from a base model to the model with tailoring layers.
        Note: CN layers are not modified (likely will be ones and zeros).
    '''
    model_dict = model.state_dict()
    base_params = {k: v for k, v in base_state_dict.items(
    ) if k in model_dict and 'gamma' not in k and 'beta' not in k}
    model_dict.update(base_params)
    model.load_state_dict(model_dict)


def compare_models(state_dict1, state_dict2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(state_dict1.items(), state_dict2.items()):
        if not torch.equal(key_item_1[1], key_item_2[1]):
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Model state dicts match perfectly!')


def batch_norm_to_identity(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm
    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = nn.Identity()
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_identity(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer


def batch_norm_to_group_norm(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm
    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    # TODO: logic for multiple groups
                    layer._modules[name] = nn.GroupNorm(
                        1, sub_layer.num_features)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_group_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer


def random_init_weights(module):
    for m in module.modules():
        try:
            m.reset_parameters()
        except:
            pass
        try:
            m.reset_running_stats()
        except:
            pass


def modernize_model(old_model_path, opt):
    # converts SVG model to a modern pytorch version
    if opt.image_width == 128:
        print('using vgg_128')
        import models.vgg_128 as model
    else:
        print('using vgg_64')
        import models.vgg_64 as model
    tmp = torch.load(old_model_path)
    frame_predictor = tmp['frame_predictor']
    posterior = tmp['posterior']
    prior = tmp['prior']
    frame_predictor.eval()
    prior.eval()
    posterior.eval()

    # Encoder and decoder have CN layers and the weights of the `opt.model_path` model
    encoder = model.encoder(
        tmp['encoder'].dim, opt.channels, use_cn_layers=True, batch_size=opt.batch_size)
    load_base_weights(encoder, tmp['encoder'].state_dict())

    decoder = model.decoder(
        tmp['decoder'].dim, opt.channels, use_cn_layers=True, batch_size=opt.batch_size)
    load_base_weights(decoder, tmp['decoder'].state_dict())

    replace_cn_layers(encoder, batch_size=opt.batch_size)
    replace_cn_layers(decoder, batch_size=opt.batch_size)
    encoder.eval()
    decoder.eval()
    frame_predictor.batch_size = opt.batch_size
    posterior.batch_size = opt.batch_size
    prior.batch_size = opt.batch_size

#     frame_predictor.cuda()
#     posterior.cuda()
#     prior.cuda()
#     encoder.cuda()
#     decoder.cuda()

    def recursion_change_bn(module):
        if isinstance(module, torch.nn.UpsamplingNearest2d):
            module.align_corners = None
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = True
            module.register_buffer('num_batches_tracked',
                                   torch.tensor(0, dtype=torch.long))
        else:
            for i, (name, module1) in enumerate(module._modules.items()):
                module1 = recursion_change_bn(module1)
        return module

    for i, (name, module) in enumerate(frame_predictor._modules.items()):
        module = recursion_change_bn(module)
    for i, (name, module) in enumerate(posterior._modules.items()):
        module = recursion_change_bn(module)
    for i, (name, module) in enumerate(prior._modules.items()):
        module = recursion_change_bn(module)
    for i, (name, module) in enumerate(encoder._modules.items()):
        module = recursion_change_bn(module)
    for i, (name, module) in enumerate(decoder._modules.items()):
        module = recursion_change_bn(module)

    if hasattr(opt, 'encoder_emb') and opt.encoder_emb:
        print('\tEncoderEmbedding')
        embedding = EncoderEmbedding(encoder, opt.emb_dim, opt.image_width,
                                     nc=opt.channels, num_emb_frames=opt.num_emb_frames,
                                     batch_size=opt.batch_size)
    else:
        print('\tConservedEmbedding')
        embedding = ConservedEmbedding(emb_dim=opt.emb_dim, image_width=opt.image_width,
                                       nc=opt.num_emb_frames * opt.channels)
    return SVGModel(encoder, frame_predictor, decoder, prior, posterior, embedding).cuda()

def compute_losses(gen_seq, batch, mus, logvars, mu_ps, logvar_ps, embedding, true_pde_embedding, params, opt):
    if opt.learned_pinn_loss and opt.pinn_outer_loss:
        # final_mse_loss, final_pde_loss, torch.stack(avg_pde_residual).sum(), data_loss, relative_data_loss
        opt_outer_mse_loss, outer_pde_loss,pde_residual, outer_mse_loss, outer_relative_loss, outer_inner_log_loss = svg_crit(
            gen_seq, batch, mus, logvars, mu_ps, logvar_ps, embedding, params, opt)
        
    else:
        # final_mse_loss, final_pde_loss, torch.stack(avg_pde_residual).sum(), data_loss, relative_data_loss
        opt_outer_mse_loss, outer_pde_loss,pde_residual, outer_mse_loss, outer_relative_loss, outer_inner_log_loss = svg_crit(
            gen_seq, batch, mus, logvars, mu_ps, logvar_ps, true_pde_embedding, params, opt)
        opt_outer_mse_loss = opt_outer_mse_loss.mean()
        outer_relative_loss = outer_relative_loss.mean()
        outer_mse_loss = outer_mse_loss.mean()
        outer_pde_loss = outer_pde_loss.mean()
    if opt.no_data_loss and opt.pinn_outer_loss:
        outer_loss = outer_pde_loss #total data + PDE loss                
    else:
        outer_loss = opt_outer_mse_loss + outer_pde_loss
    
    return outer_loss, outer_pde_loss, outer_mse_loss, pde_residual, outer_relative_loss, outer_inner_log_loss

def plot_solution_field(pre_tailor_output, post_tailor_output, ground_truth, opt, params, epoch,save_dir):
    if '1d' in opt.dataset and epoch % 20 == 0:
        param = params[0].item()
        plt.figure()
        plt.plot(pre_tailor_output[-1].view(-1).detach().cpu().numpy(), label = 'pre_tailor_output')
        plt.plot(post_tailor_output[-1].view(-1).detach().cpu().numpy(), label = 'post_tailor_output')
        plt.plot(ground_truth[-1].view(-1).detach().cpu().numpy(), label = 'ground_truth')
        plt.yscale('log')
        plt.title(f'param {param} epoch {epoch}')
        plt.legend()
        plt.show()
        plt.savefig(save_dir + f'/param {param} epoch {epoch}.png')

    if '2d' in opt.dataset and epoch % 20 == 0:
        # 
        plt.figure()
        param = params[0].item()
        plt.imshow(pre_tailor_output[-1][0,0].detach().cpu().numpy(), label = 'pre_tailor_output')
        plt.title(f'pre_tailor_output param {param} epoch {epoch}')
        plt.legend()
        plt.show()
        plt.savefig(save_dir + f'/pre_tailor_output param {param} epoch {epoch}.png')
        
        plt.figure()
        plt.imshow(post_tailor_output[-1][0,0].detach().cpu().numpy(), label = 'post_tailor_output')
        plt.title(f'post_tailor_output param {param} epoch {epoch}')
        plt.legend()
        plt.show()
        plt.savefig(save_dir + f'/post_tailor_output param {param} epoch {epoch}.png')

        plt.figure()
        plt.imshow(ground_truth[-1][0,0].detach().cpu().numpy(), label = 'ground_truth')
        plt.title(f'ground_truth param {param} epoch {epoch}')
        plt.legend()
        plt.show()
        plt.savefig(save_dir + f'/ground_truth param {param} epoch {epoch}.png')

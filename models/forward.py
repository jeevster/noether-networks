import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import utils
import numpy as np
import copy
import contextlib
import pdb
import higher
from torchviz import make_dot

from models.cn import replace_cn_layers, load_cached_cn_modules, cache_cn_modules
from utils import svg_crit, compute_losses
import copy

def inner_crit(fmodel, gen_seq, true_params, mode='mse', num_emb_frames=1, compare_to='prev', setting='train', opt=None, emb_mode = 'emb',
               learnable_model = None):
    # compute embeddings for sequence
    # pdb.set_trace()
    param_loss = []
    embs = []
    if num_emb_frames == 1:
        val_nu_loss = 0
        for frame in gen_seq:
            pde_value, true_pde_value, pred_params = fmodel(frame, true_params = true_params, mode=emb_mode, return_params = True)
            nu_pred = pred_params[0]
            nu = true_params[0]
            val_nu_loss += ((nu_pred - nu).abs() / nu).mean()
            embs.append(pde_value)
            param_loss.append(val_nu_loss)
    elif num_emb_frames >1:
        assert(len(gen_seq) >= num_emb_frames)
        stacked_gen_seq = []
        for i in range(num_emb_frames, len(gen_seq)+1):
            stacked_gen_seq.append(
                torch.stack([g for g in gen_seq[i-num_emb_frames:i]], dim=1))
        val_nu_loss = 0
        for frame in stacked_gen_seq:
            pde_value, true_pde_value, pred_params = fmodel(frame, true_params = true_params, mode=emb_mode, return_params = True)#, add_learnable = True)

            nu_pred = pred_params[0].to(torch.cuda.current_device())
            nu = true_params[0].to(torch.cuda.current_device())
            val_nu_loss += ((nu_pred - nu).abs() / nu).mean()
            if learnable_model != None:
                if opt.channels == 1:
                    x_hat = learnable_model(frame[:,:,0,:,0])
                    pde_value = pde_value + x_hat[:,0,0]
                else:
                    x_hat = learnable_model(frame[:,:,:,:,:])
                    pde_value = pde_value + x_hat[:,0]
            embs.append(pde_value)
            param_loss.append(val_nu_loss)
        assert(len(embs) == len(gen_seq) - num_emb_frames + 1)  
    else:
        raise ValueError

    if setting == 'eval':
        val_inner_lr = opt.inner_lr
        if opt.val_inner_lr != -1:
            val_inner_lr = opt.val_inner_lr
        _embs = torch.stack([i.detach() for i in embs])
        experiment_id = opt.model_path.split('/')[-2]
        baseline_fname = f'eval_metrics/genseq_{experiment_id}'
        if val_inner_lr > 0 and opt.num_inner_steps > 0:
            baseline_fname += f'-lr{val_inner_lr}'
            baseline_fname += f'-steps{opt.num_inner_steps}'
        baseline_fname += '.npy'
        _gen_seq = gen_seq
        if torch.is_tensor(_gen_seq):
            _gen_seq = _gen_seq.detach().cpu().numpy()
        elif isinstance(_gen_seq, list):
            _gen_seq = torch.stack([i.detach()
                                    for i in _gen_seq]).cpu().numpy()
        np.save(baseline_fname, np.asarray(_gen_seq))

        baseline_fname = f'eval_metrics/embeddings_{experiment_id}'
        if val_inner_lr > 0 and opt.num_inner_steps > 0:
            baseline_fname += f'-lr{val_inner_lr}'
            baseline_fname += f'-steps{opt.num_inner_steps}'
        baseline_fname += '.npy'
        np.save(baseline_fname, _embs.cpu().numpy())
    
    if mode == 'mse':
        # we penalize the pairwise losses
        if compare_to == 'prev':
            pairwise_inner_losses = torch.stack([F.mse_loss(
                embs[t-1], embs[t], reduction='none') for t in range(1, len(embs))]).mean(dim=2)
        elif compare_to == 'zero':
            pairwise_inner_losses = torch.stack([F.mse_loss(
                embs[0], embs[t], reduction='none') for t in range(1, len(embs))]).mean(dim=2)
        elif compare_to == 'zero_and_prev':
            pairwise_inner_losses = torch.stack([F.mse_loss(embs[t-1], embs[t], reduction='none') for t in range(
                1, len(embs))] + [F.mse_loss(embs[0], embs[t], reduction='none') for t in range(1, len(embs))]).mean(dim=2)
        elif compare_to == 'pde_zero':
            pairwise_inner_losses = torch.stack(
                [torch.square(emb).mean() for emb in embs])
        elif compare_to == 'pde_log':
            pairwise_inner_losses = torch.stack(
                [torch.abs(emb).log10().mean() for emb in embs])
        else:
            raise ValueError('inner_crit_compare_to must be one of [prev, zero, zero_and_prev, pde_zero, or pde_log')
    elif mode == 'cosine':
        # cosine distance is 1 minus cosine similarity
        pairwise_inner_losses = torch.stack(
            [1 - F.cosine_similarity(embs[t-1], embs[t]) for t in range(1, len(embs))])
    else:
        raise NotImplementedError('please use either "mse" or "cosine" mode')
    # total inner loss is just the sum of pairwise losses
    pairwise_abs_inner_losses = torch.stack([torch.abs(emb).mean() for emb in embs])
    return torch.sum(pairwise_inner_losses, dim=0), torch.sum(torch.stack(param_loss), dim=0), torch.sum(pairwise_abs_inner_losses, dim = 0)


def predict_many_steps(func_model, gt_seq, true_params, opt, mode='eval', prior_epses=[], posterior_epses=[], learnable_model = False, phi_hat = None):
    mus, logvars, mu_ps, logvar_ps = [], [], [], []
    if 'Basic' not in type(func_model).__name__:
        if getattr(func_model.frame_predictor, 'init_hidden', None) is not None:
            func_model.frame_predictor.hidden = func_model.frame_predictor.init_hidden()
        # func_model.posterior.hidden = func_model.posterior.init_hidden()
        # func_model.prior.hidden = func_model.prior.init_hidden()

#     print(f'predict_many_steps - prior_epses: {prior_epses}')
    # pdb.set_trace()
    #initial condition - condition on n_past frames
    gen_seq = gt_seq[0:opt.n_past]

    # skip connections for this prediction sequence: always take the latest GT one
    #     (after opt.n_past time steps, this will be the last GT frame)
    skip = [None]

    for i in range(opt.n_past, int(opt.n_eval/opt.frame_step)):
        # TODO: different mode for training, where we get frames for more than just conditioning?
        if mode == 'eval':
            gt = None if i >= opt.n_eval else gt_seq[i]
            # TODO: the following line causes issues, is there an elegant way to do stop grad?
            # x_in = gen_seq[-1].clone().detach()
            # this one seems to work, but is super hacky
            if hasattr(opt, 'stop_grad') and opt.stop_grad:
                x_in = torch.cat(gen_seq[-opt.n_past:], dim =1).clone().detach()
            else:
                # and this one doesn't do stop grad at all
                # condition on last n_past generated frames
                # pdb.set_trace()
                if opt.conditioning:
                    tiled_seq = [torch.cat([field, phi_hat],dim = 1) for field in gen_seq[-opt.n_past:]]
                    x_in = torch.cat(tiled_seq, dim = 1)
                else:
                    x_in = torch.cat(gen_seq[-opt.n_past:], dim =1)

        elif mode == 'train':
            gt = gt_seq[i]
            #condition on last n_past ground truth frames (teacher forcing)
            if opt.conditioning:
                tiled_seq = [torch.cat([field, phi_hat],dim = 1) for field in gt_seq[(i-opt.n_past):i]]
                x_in = torch.cat(tiled_seq, dim = 1)
            else:
                x_in = torch.cat(gt_seq[(i-opt.n_past):i], dim = 1)
        else:
            raise NotImplementedError
#         gt = None if i >= opt.n_past else gt_seq[i]

        
        prior_eps = [None]
        posterior_eps = [None]
#         print(f"i: {i}")
        if i-1 < len(prior_epses) and i-1 < len(posterior_epses):
            prior_eps = prior_epses[i-1]
            posterior_eps = posterior_epses[i-1]
#             print('re-using eps')
#         else:
#             print('sampling eps')

        # predict
        x_hat, mu, logvar, mu_p, logvar_p, skip = func_model(
            x_in,
            gt, true_params, skip, opt,
            i=i, mode=mode,
            prior_eps=prior_eps,
            posterior_eps=posterior_eps,
        )
#         print(f'prior_eps[0,0] after func_model:  {prior_eps[0][0,0]}')

        if not (i-1 < len(prior_epses) and i-1 < len(posterior_epses)):
            #            print('appending to lstm_eps')
            # prior_epses.append([prior_eps[0].detach()])
            # posterior_epses.append([posterior_eps[0].detach()])
            pass
            
        #shouldn't get executed anymore because of loop boundaries
        if i < opt.n_past:
            gen_seq.append(gt_seq[i])
        else:
            gen_seq.append(x_hat)
        # track statistics from prior and posterior for KL divergence loss term
        mus.append(mu)
        logvars.append(logvar)
        mu_ps.append(mu_p)
        logvar_ps.append(logvar_p)

    return gen_seq, mus, logvars, mu_ps, logvar_ps


def tailor_many_steps(svg_model, x, true_pde_embedding, params, opt, track_higher_grads=True, mode='eval',learnable_model = None, **kwargs):
    '''
    Perform a round of tailoring.
    '''
    # pdb.set_trace()
    if not hasattr(opt, 'num_emb_frames'):
        opt.num_emb_frames = 1  # number of frames to pass to the embedding
    # re-initialize CN params
    # TODO: uncomment
    replace_cn_layers(svg_model.encoder)
    replace_cn_layers(svg_model.decoder)
    replace_cn_layers(svg_model.frame_predictor)
    if 'load_cached_cn' in kwargs and kwargs['load_cached_cn'] and \
            'cached_cn' in kwargs and kwargs['cached_cn'][0] is not None:
        load_cached_cn_modules(svg_model, kwargs['cached_cn'][0])
    # TODO: investigate the effect of not replacing these after jump step zero
    # _cn_beta = list(filter(lambda p: 'beta' in p[0], svg_model.decoder.named_parameters()))
    # print(f'CN layer gamma: {_cn_beta[1]}')

    cn_module_params = list(svg_model.decoder.named_parameters())
    if not 'only_cn_decoder' in kwargs or not kwargs['only_cn_decoder']:
        cn_module_params += list(svg_model.encoder.named_parameters())

    cn_params = [p[1] for p in cn_module_params if ('gamma' in p[0] or 'beta' in p[0])]

    if 'Basic' in type(svg_model).__name__:
        cn_params = list(svg_model.encoder.parameters()) + list(svg_model.decoder.parameters())

    # elif opt.inner_opt_all_model_weights:
        # TODO: try with ALL modules, not just enc and dec
        # cn_params = list(svg_model.encoder.parameters()) + list(svg_model.decoder.parameters()) \
            # + list(svg_model.prior.parameters()) + list(svg_model.posterior.parameters()) + \
                # list(svg_model.frame_predictor.parameters())

    frame_predictor_params =  list(svg_model.frame_predictor.named_parameters())
    if opt.use_cn == True and opt.inner_opt_all_model_weights == False:
        cn_params += [p[1] for p in frame_predictor_params if ('gamma' in p[0] or 'beta' in p[0])]
    elif opt.use_cn == False and opt.inner_opt_all_model_weights == True:

        cn_params += [p[1] for p in frame_predictor_params if not ('gamma' in p[0] or 'beta' in p[0])]
    elif opt.use_cn == True and opt.inner_opt_all_model_weights == True:
        cn_params += list(svg_model.frame_predictor.parameters())

    inner_lr = opt.inner_lr
    if 'val_inner_lr' in kwargs:
        inner_lr = kwargs['val_inner_lr']

    if not hasattr(opt, 'inner_crit_compare_to'):
        opt.inner_crit_compare_to = 'prev'

    inner_opt = optim.SGD(cn_params, lr=inner_lr)
    if 'adam_inner_opt' in kwargs and kwargs['adam_inner_opt']:
        inner_opt = optim.Adam(cn_params, lr=inner_lr)

    inner_crit_mode = 'mse'
    if 'inner_crit_mode' in kwargs:
        inner_crit_mode = kwargs['inner_crit_mode']

    # tailor_loss_gain = []
    # true_tailor_loss_gain = []
    tailor_losses = []
    tailor_abs_losses = []
    true_tailor_losses = []
    true_tailor_abs_losses = []
    param_losses = []
    true_param_losses = []
    svg_losses = []
    ssims = []
    psnrs = []
    mses = []
    epsilons = []
    data_loss_collector = []
    orig_gen_seq = None
    orig_tailor_loss = None
    cn_tailored_params = None
    normal_tailored_params = None
    cache = cache_cn_modules(svg_model)
    try:
        cn_original_params = cache['frame_predictor.CNlayers.0.gamma']
        normal_original_params =  cache['frame_predictor.lifting.fc.weight']
    except:
        cn_original_params = 0
        normal_original_params = 0
    with higher.innerloop_ctx(
        svg_model,
        inner_opt,
        track_higher_grads=track_higher_grads,
        copy_initial_weights=False
        # copy_initial_weights=True  # grads are zero if we copy initial weights!!
    ) as (fmodel, diffopt):
        prior_epses = []
        posterior_epses = []
        loss_collector = []
        true_loss_collector = []
        abs_loss_collector = []
        true_abs_loss_collector = []

        # TODO: set requires_grad=False for the outer params
        if opt.tailor == True:
            print("TAILORING")
            for inner_step in range(opt.num_inner_steps):
                if 'reuse_lstm_eps' not in kwargs or not kwargs['reuse_lstm_eps']:
                    # print('not re-use lstm eps')
                    prior_epses = []
                    posterior_epses = []
                # inner step: make a prediction, compute inner loss, backprop wrt inner loss

                # print(f'beginning of step {inner_step} of tailoring loop: prior_epses = {prior_epses}')

                # autoregressive rollout
                gen_seq, mus, logvars, mu_ps, logvar_ps = predict_many_steps(fmodel, x, params, opt, mode=mode,
                                                                             prior_epses=prior_epses,
                                                                             posterior_epses=posterior_epses,
                                                                             learnable_model = learnable_model,
                                                                             phi_hat = kwargs['phi_hat']
                                                                             )
                # compute Noether loss
                tailor_loss, param_loss, tailor_abs_loss = inner_crit(fmodel, gen_seq, params, mode=inner_crit_mode,
                                                    num_emb_frames=opt.num_emb_frames,learnable_model = learnable_model,
                                                    compare_to=opt.inner_crit_compare_to, setting=mode, opt=opt)
                loss_collector.append(tailor_loss)
                abs_loss_collector.append(tailor_abs_loss)
                #compute true PDE loss if using learnable embedding
                # true_tailor_loss = None
                # if opt.emb_type != 'pde_const_emb':
                with torch.no_grad():
                    _, outer_mse_loss, _, _, _ = compute_losses(gen_seq, x, mus, logvars, mu_ps, logvar_ps, true_pde_embedding, true_pde_embedding, params, opt)
                    true_tailor_loss, true_param_loss, true_tailor_abs_loss = inner_crit(fmodel, gen_seq, params, mode=inner_crit_mode,
                                                                                    num_emb_frames=opt.num_emb_frames,learnable_model = learnable_model,
                                                                                    compare_to=opt.inner_crit_compare_to, setting=mode, 
                                                                                    opt=opt, emb_mode = 'true_emb')
                    true_loss_collector.append(true_tailor_loss)
                    data_loss_collector.append(outer_mse_loss)
                    true_abs_loss_collector.append(true_tailor_abs_loss)

                if inner_step == 0:
                    # print('writing orig_gen_seq and orig_tailor_loss')
                    orig_gen_seq = [f.detach() for f in gen_seq]
                    orig_tailor_loss = tailor_loss.detach()

                loss = tailor_loss.mean()

                # don't do this (as of now)
                if opt.learn_inner_lr:
                    # we optionally meta-learn the inner learning rate (log scale)
                    loss *= torch.exp(list(filter(lambda x: x.size() == torch.Size([]),
                                                  [param for param in svg_model.parameters()]))[0])

                # gradient tailoring step on Noether loss
                diffopt.step(loss)
                cache = cache_cn_modules(fmodel)
                cn_tailored_params = cache['frame_predictor.CNlayers.0.gamma']
                # with torch.no_grad():
                normal_tailored_params = cache['frame_predictor.lifting.fc.weight']

                # cache CN params
                if 'cached_cn' in kwargs:
                    kwargs['cached_cn'][0] = cache_cn_modules(fmodel)

                # TODO: test outer opt pass in inner loop

                # don't do this as of now
                if 'svg_crit' in kwargs:
                    outer_loss = kwargs['svg_crit'](
                        gen_seq, x, mus, logvars, mu_ps, logvar_ps, opt).mean()
                    # print(f'outer_loss in inner loop: {outer_loss}')
                    outer_loss.backward()

                # track metrics
                # TODO: also compute outer loss at each step for plotting
                tailor_losses.append(tailor_loss.detach().mean().item())
                tailor_abs_losses.append(tailor_abs_loss.detach().mean().item())
                param_losses.append(param_loss.detach().mean().item())
                if true_tailor_loss is not None:
                    true_tailor_losses.append(true_tailor_loss.detach().mean().item())
                    true_tailor_abs_losses.append(true_tailor_abs_loss.detach().mean().item())
                    true_param_losses.append(true_param_loss.detach().mean().item())

                if 'tailor_ssims' in kwargs:
                    # compute SSIM for gen_seq batch
                    mse, ssim, psnr = utils.eval_seq([f.detach() for f in x[opt.n_past:]],
                                                     [f.detach() for f in gen_seq[opt.n_past:]])
                    ssims.append(ssim)
                    psnrs.append(psnr)
                    mses.append(mse)

                svg_mse_loss, svg_pde_loss,_ = svg_crit(gen_seq, x, mus, logvars,
                                    mu_ps, logvar_ps, true_pde_embedding, params, opt)
                svg_mse_loss = svg_mse_loss.detach().cpu().item()
                svg_pde_loss = svg_pde_loss.detach().cpu().item()
                svg_losses.append(svg_mse_loss) #only keep data loss for plotting

                # # TODO: remove next two lines
                # _cn_beta = list(filter(lambda p: 'beta' in p[0], fmodel.decoder.named_parameters()))
                # print(f'CN layer beta: {_cn_beta[1]}')

                if 'reuse_lstm_eps' not in kwargs or not kwargs['reuse_lstm_eps']:
                    # print('not re-use lstm eps')
                    prior_epses = []
                    posterior_epses = []

        # generate the final model prediction with the tailored weights
        final_gen_seq, mus, logvars, mu_ps, logvar_ps = predict_many_steps(fmodel, x, params, opt, mode=mode,
                                                                            prior_epses=prior_epses,
                                                                            posterior_epses=posterior_epses,
                                                                            learnable_model = learnable_model,
                                                                            phi_hat = kwargs['phi_hat'] if 'phi_hat' in kwargs else None
                                                                        )

        # track metrics
        # if opt.tailor:
        #want to measure PDE residual loss even when not tailoring
        with torch.no_grad():
            _, outer_mse_loss, _, _, _ = compute_losses(final_gen_seq, x, mus, logvars, mu_ps, logvar_ps, true_pde_embedding, true_pde_embedding, params, opt)
            data_loss_collector.append(outer_mse_loss)
            tailor_loss, param_loss, tailor_abs_loss = inner_crit(fmodel, final_gen_seq, params, mode=inner_crit_mode,
                                    num_emb_frames=opt.num_emb_frames,
                                    opt=opt,learnable_model = learnable_model,
                                    compare_to=opt.inner_crit_compare_to)#.detach()
            loss_collector.append(tailor_loss)
            abs_loss_collector.append(tailor_abs_loss)
            tailor_losses.append(tailor_loss.detach().mean().cpu().item())
            tailor_abs_losses.append(tailor_abs_loss.detach().mean().cpu().item())
            param_losses.append(param_loss.detach().mean().cpu().item())

# if opt.tailor:# and opt.emb_type != 'pde_const_emb':
        with torch.no_grad():
            true_tailor_loss, true_param_loss, true_tailor_abs_loss = inner_crit(fmodel, final_gen_seq, params, mode=inner_crit_mode,
                                    num_emb_frames=opt.num_emb_frames,
                                    opt=opt,learnable_model = learnable_model,
                                    compare_to=opt.inner_crit_compare_to, emb_mode = 'true_emb')#.detach()
            true_tailor_losses.append(true_tailor_loss.detach().mean().cpu().item())
            true_tailor_abs_losses.append(true_tailor_abs_loss.detach().mean().cpu().item())
            true_param_losses.append(true_param_loss.detach().mean().cpu().item())
            true_loss_collector.append(true_tailor_loss)
            true_abs_loss_collector.append(true_tailor_abs_loss)

        svg_mse_loss, svg_pde_loss,_ = svg_crit(final_gen_seq, x, mus, logvars,
                            mu_ps, logvar_ps, true_pde_embedding, params, opt)
        svg_mse_loss = svg_mse_loss.detach().cpu().item()
        svg_pde_loss = svg_pde_loss.detach().cpu().item()
        svg_losses.append(svg_mse_loss) #only keep the data loss for logging

        if 'tailor_ssims' in kwargs:
            # compute SSIM for gen_seq batch
            mse, ssim, psnr = utils.eval_seq([f.detach() for f in x[opt.n_past:]],
                                             [f.detach() for f in final_gen_seq[opt.n_past:]])
            ssims.append(ssim)
            psnrs.append(psnr)
            mses.append(mse)
        # I think this isn't actually being run but need to double check TODO
        if opt.only_tailor_on_improvement and orig_gen_seq is not None and orig_tailor_loss is not None and opt.tailor:

            #             print(f'orig_tailor_loss > tailor_loss: {orig_tailor_loss > tailor_loss}')

            # per-batch basis
            #             final_gen_seq = orig_gen_seq
            # per-sequence basis
            #             print(f'fin.shape: {final_gen_seq[0].shape}')
            mask = (orig_tailor_loss > tailor_loss).detach().view(-1, 1, 1, 1)
            print(
                f'percent of sequences in batch improved by tailoring: {mask.float().mean()}')
#             print(f'mask shape: {mask.shape}')

            final_gen_seq = [torch.where(mask, fin, orig)
                             for fin, orig in zip(final_gen_seq, orig_gen_seq)]

            svg_mse_loss, svg_pde_loss,_ = svg_crit(final_gen_seq, x, mus, logvars,
                                mu_ps, logvar_ps, true_pde_embedding, params, opt)
            svg_mse_loss = svg_mse_loss.detach().cpu().item()
            svg_pde_loss = svg_pde_loss.detach().cpu().item()
            svg_losses.append(svg_mse_loss) #only keep the data loss for logging

            with torch.no_grad():


                tailor_loss = inner_crit(fmodel, final_gen_seq, params, mode=inner_crit_mode,
                                     num_emb_frames=opt.num_emb_frames, opt=opt,learnable_model = learnable_model,
                                     compare_to=opt.inner_crit_compare_to).detach()
            # if opt.emb_type != 'pde_const_emb':
                true_tailor_loss = inner_crit(fmodel, final_gen_seq, params, mode=inner_crit_mode,
                                    num_emb_frames=opt.num_emb_frames, opt=opt,learnable_model = learnable_model,
                                    compare_to=opt.inner_crit_compare_to, emb_mode = 'true_emb').detach()
            true_tailor_losses.append(true_tailor_loss.mean().detach().cpu().item())
            tailor_losses.append(tailor_loss.mean().detach().cpu().item())
            # inner_gain.append()
            # true_inner_gain.append()
            if 'tailor_ssims' in kwargs:
                # compute SSIM for gen_seq batch
                mse, ssim, psnr = utils.eval_seq([f.detach() for f in x[opt.n_past:]],
                                                 [f.detach() for f in final_gen_seq[opt.n_past:]])
                ssims.append(ssim)
                psnrs.append(psnr)
                mses.append(mse)

    # print(f'    avg INNER losses: {sum(tailor_losses) / len(tailor_losses)}')
    # track metrics
    # pdb.set_trace()
    if 'cn_norm_tracker' in kwargs:
        if cn_tailored_params != None:
            kwargs['cn_norm_tracker'][0].append(torch.norm(cn_original_params).detach().cpu().item())
            kwargs['cn_norm_tracker'][1].append(torch.norm(cn_tailored_params).detach().cpu().item())
            kwargs['cn_norm_tracker'][2].append(torch.norm(cn_tailored_params - cn_original_params).detach().cpu().item())
        else:
            kwargs['cn_norm_tracker'][0].append(torch.norm(cn_original_params).detach().cpu().item())
            kwargs['cn_norm_tracker'][1].append(torch.norm(cn_original_params).detach().cpu().item())
            kwargs['cn_norm_tracker'][2].append(torch.norm(cn_original_params - cn_original_params).detach().cpu().item())
    if 'normal_norm_tracker' in kwargs:
        if normal_tailored_params != None:
            kwargs['normal_norm_tracker'][0].append(torch.norm(normal_original_params).detach().cpu().item())
            kwargs['normal_norm_tracker'][1].append(torch.norm(normal_tailored_params).detach().cpu().item())
            kwargs['normal_norm_tracker'][2].append(torch.norm(normal_tailored_params - normal_original_params).detach().cpu().item())
        else:
            kwargs['normal_norm_tracker'][0].append(torch.norm(normal_original_params).detach().cpu().item())
            kwargs['normal_norm_tracker'][1].append(torch.norm(normal_original_params).detach().cpu().item())
            kwargs['normal_norm_tracker'][2].append(torch.norm(normal_original_params - normal_original_params).detach().cpu().item())
    if 'tailor_losses' in kwargs:
        kwargs['tailor_losses'].append(tailor_losses)
    if 'tailor_abs_losses' in kwargs:
        kwargs['tailor_abs_losses'].append(tailor_abs_losses)
    if 'inner_gain' in kwargs:
        difference = (loss_collector[-1] - loss_collector[0]) / loss_collector[0]
        kwargs['inner_gain'].append(difference.mean().detach().cpu().item())
    if 'abs_inner_gain' in kwargs:
        difference = (abs_loss_collector[-1] - abs_loss_collector[0]) / abs_loss_collector[0]
        kwargs['abs_inner_gain'].append(difference.mean().detach().cpu().item())
    if 'data_gain' in kwargs:
        difference =( data_loss_collector[-1] - data_loss_collector[0]) / data_loss_collector[0]
        kwargs['data_gain'].append(difference.mean().detach().cpu().item())
    if 'true_inner_gain' in kwargs:
        difference = (true_loss_collector[-1] - true_loss_collector[0]) / true_loss_collector[0]
        kwargs['true_inner_gain'].append(difference.mean().detach().cpu().item())
    if 'true_abs_inner_gain' in kwargs:
        difference = (true_abs_loss_collector[-1] - true_abs_loss_collector[0]) / true_abs_loss_collector[0]
        kwargs['true_abs_inner_gain'].append(difference.mean().detach().cpu().item())
    if 'true_tailor_losses' in kwargs:# in kwargs and opt.tailor and opt.emb_type != 'pde_const_emb':
        kwargs['true_tailor_losses'].append(true_tailor_losses)
    if 'true_tailor_abs_losses' in kwargs:# in kwargs and opt.tailor and opt.emb_type != 'pde_const_emb':
        kwargs['true_tailor_abs_losses'].append(true_tailor_abs_losses)
    if 'param_losses' in kwargs:
        kwargs['param_losses'].append(param_losses)
    if 'true_param_losses' in kwargs:
        kwargs['true_param_losses'].append(true_param_losses)
    if all(m in kwargs for m in ('tailor_ssims', 'tailor_psnrs', 'tailor_mses')):
        kwargs['tailor_ssims'].append(ssims)
        kwargs['tailor_psnrs'].append(psnrs)
        kwargs['tailor_mses'].append(mses)
    if 'svg_losses' in kwargs:
        kwargs['svg_losses'].append(svg_losses)

    # we need the first and second order statistics of the posterior and prior for outer (SVG) loss
    return final_gen_seq, mus, logvars, mu_ps, logvar_ps


def dont_tailor_many_steps(svg_model, x, true_pde_embedding, params, opt, track_higher_grads=True, mode='eval',learnable_model = None, **kwargs):
    '''
    simple predictions
    '''
    # pdb.set_trace()
    if not hasattr(opt, 'num_emb_frames'):
        opt.num_emb_frames = 1  # number of frames to pass to the embedding
    # re-initialize CN params
    # TODO: uncomment
    replace_cn_layers(svg_model.encoder)
    replace_cn_layers(svg_model.decoder)
    if 'load_cached_cn' in kwargs and kwargs['load_cached_cn'] and \
            'cached_cn' in kwargs and kwargs['cached_cn'][0] is not None:
        load_cached_cn_modules(svg_model, kwargs['cached_cn'][0])
    # TODO: investigate the effect of not replacing these after jump step zero
    # _cn_beta = list(filter(lambda p: 'beta' in p[0], svg_model.decoder.named_parameters()))
    # print(f'CN layer gamma: {_cn_beta[1]}')

    cn_module_params = list(svg_model.decoder.named_parameters())
    if not 'only_cn_decoder' in kwargs or not kwargs['only_cn_decoder']:
        cn_module_params += list(svg_model.encoder.named_parameters())
    cn_params = [p[1] for p in cn_module_params if (
        'gamma' in p[0] or 'beta' in p[0])]

    if 'Basic' in type(svg_model).__name__:
        cn_params = list(svg_model.encoder.parameters()) + \
            list(svg_model.decoder.parameters())

    elif opt.inner_opt_all_model_weights:
        # TODO: try with ALL modules, not just enc and dec
        cn_params = list(svg_model.encoder.parameters()) + list(svg_model.decoder.parameters()) \
            + list(svg_model.prior.parameters()) + list(svg_model.posterior.parameters()) + \
            list(svg_model.frame_predictor.parameters())

    inner_lr = opt.inner_lr
    if 'val_inner_lr' in kwargs:
        inner_lr = kwargs['val_inner_lr']

    if not hasattr(opt, 'inner_crit_compare_to'):
        opt.inner_crit_compare_to = 'prev'

    inner_crit_mode = 'mse'
    if 'inner_crit_mode' in kwargs:
        inner_crit_mode = kwargs['inner_crit_mode']

    tailor_losses = []
    true_tailor_losses = []
    param_losses = []
    true_param_losses = []
    svg_losses = []
    ssims = []
    psnrs = []
    mses = []

    prior_epses = []
    posterior_epses = []
    loss_collector = []
    true_loss_collector = []

    final_gen_seq, mus, logvars, mu_ps, logvar_ps = predict_many_steps(svg_model, x, params, opt, mode=mode,
                                                                        prior_epses=prior_epses,
                                                                        posterior_epses=posterior_epses,
                                                                        learnable_model = learnable_model
                                                                        )

    # track metrics
    # if opt.tailor:
    #want to measure PDE residual loss even when not tailoring
    with torch.no_grad():
        tailor_loss, param_loss, _ = inner_crit(svg_model, final_gen_seq, params, mode=inner_crit_mode,
                                num_emb_frames=opt.num_emb_frames,
                                opt=opt,learnable_model = learnable_model,
                                compare_to=opt.inner_crit_compare_to)#.detach()
        loss_collector.append(tailor_loss)
        tailor_losses.append(tailor_loss.detach().mean().cpu().item())
        param_losses.append(param_loss.detach().mean().cpu().item())

    # if opt.tailor:# and opt.emb_type != 'pde_const_emb':
    with torch.no_grad():
        true_tailor_loss, true_param_loss, _ = inner_crit(svg_model, final_gen_seq, params, mode=inner_crit_mode,
                                num_emb_frames=opt.num_emb_frames,
                                opt=opt,learnable_model = learnable_model,
                                compare_to=opt.inner_crit_compare_to, emb_mode = 'true_emb')#.detach()
        true_tailor_losses.append(true_tailor_loss.detach().mean().cpu().item())
        true_param_losses.append(true_param_loss.detach().mean().cpu().item())
        true_loss_collector.append(true_tailor_loss)

    svg_mse_loss, svg_pde_loss,_ = svg_crit(final_gen_seq, x, mus, logvars,
                        mu_ps, logvar_ps, true_pde_embedding, params, opt)
    svg_mse_loss = svg_mse_loss.detach().cpu().item()
    svg_pde_loss = svg_pde_loss.detach().cpu().item()
    svg_losses.append(svg_mse_loss) #only keep the data loss for logging

    if 'tailor_ssims' in kwargs:
        # compute SSIM for gen_seq batch
        mse, ssim, psnr = utils.eval_seq([f.detach() for f in x[opt.n_past:]],
                                            [f.detach() for f in final_gen_seq[opt.n_past:]])
        ssims.append(ssim)
        psnrs.append(psnr)
        mses.append(mse)

    # print(f'    avg INNER losses: {sum(tailor_losses) / len(tailor_losses)}')
    # track metrics
    # pdb.set_trace()
    if 'tailor_losses' in kwargs:
        kwargs['tailor_losses'].append(tailor_losses)
    if 'inner_gain' in kwargs:
        difference = loss_collector[-1] - loss_collector[0]
        kwargs['inner_gain'].append(difference.mean().detach().cpu().item())
    if 'true_inner_gain' in kwargs:
        difference = true_loss_collector[-1] - true_loss_collector[0]
        kwargs['true_inner_gain'].append(difference.mean().detach().cpu().item())
    if 'true_tailor_losses' in kwargs:# in kwargs and opt.tailor and opt.emb_type != 'pde_const_emb':
        kwargs['true_tailor_losses'].append(true_tailor_losses)
    if 'param_losses' in kwargs:
        kwargs['param_losses'].append(param_losses)
    if 'true_param_losses' in kwargs:
        kwargs['true_param_losses'].append(true_param_losses)

    if all(m in kwargs for m in ('tailor_ssims', 'tailor_psnrs', 'tailor_mses')):
        kwargs['tailor_ssims'].append(ssims)
        kwargs['tailor_psnrs'].append(psnrs)
        kwargs['tailor_mses'].append(mses)

    if 'svg_losses' in kwargs:
        kwargs['svg_losses'].append(svg_losses)

    # we need the first and second order statistics of the posterior and prior for outer (SVG) loss
    return final_gen_seq, mus, logvars, mu_ps, logvar_ps


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
# This sets the default model weights to float64
#torch.set_default_dtype(torch.float64)  # nopep8
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
from torch.utils.data import DataLoader
import utils
from utils import svg_crit, dump_params_to_yml
import itertools
import numpy as np
import copy
import higher
from datetime import datetime
from torch.utils.tensorboard.summary import hparams
from utils import dump_params_to_yml
from models.forward import predict_many_steps, tailor_many_steps, predict_many_steps_non_meta
from models.cn import replace_cn_layers
from models.svg import SVGModel
from models.fno_models import FNOEncoder, FNODecoder
from models.embedding import ConservedEmbedding, ConvConservedEmbedding, TwoDDiffusionReactionEmbedding
from models.OneD_embeddings import OneDEmbedding
import pdb
import models.lstm as lstm_models
from models.forward import predict_many_steps
from neuraloperator.neuralop.models import FNO, FNO1d

from torchinfo import summary

import pandas as pd
# NOTE: deterministic for debugging
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--train_embedding', default=True, type=bool, help='dummy flag indicating we are training the embedding model only. DO NOT CHANGE')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--data_root', default='data',
                    help='root directory for data')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--log_dir', default='',
                    help='directory to save generations to')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=2,
                    help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10,
                    help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=0,
                    help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=100,
                    help='number of samples')
parser.add_argument('--plot_train', type=int, default=0,
                    help='if true, also predict training data')
parser.add_argument('--use_action', type=int, default=0,
                    help='if true, train action-conditional model')
parser.add_argument('--dataset', default='bair', help='dataset to train with')
parser.add_argument('--image_width', type=int, default=64,
                    help='the height / width of the input image to network')
parser.add_argument('--n_epochs', type=int, default=100,
                    help='how many epochs to train for')
parser.add_argument('--ckpt_every', type=int, default=5,
                    help='how many epochs per checkpoint save')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--num_epochs_per_val', type=int,
                    default=1, help='perform validation every _ epochs')
parser.add_argument('--tailor', action='store_true',
                    help='if true, perform tailoring')
parser.add_argument('--num_inner_steps', type=int,
                    default=1, help='how many tailoring steps?')
parser.add_argument('--num_jump_steps', type=int, default=0,
                    help='how many tailoring steps?')
parser.add_argument('--num_train_batch', type=int,
                    default=-1, help='if -1, do all of them')
parser.add_argument('--num_val_batch', type=int,
                    default=-1, help='if -1, do all of them')
parser.add_argument('--inner_lr', type=float, default=0.0001,
                    help='learning rate for inner optimizer')
parser.add_argument('--val_inner_lr', type=float, default=-1,
                    help='val. LR for inner opt (if -1, use orig.)')
parser.add_argument('--outer_lr', type=float, default=0.0001,
                    help='learning rate for outer optimizer')
parser.add_argument('--svg_loss_kl_weight', type=float,
                    default=0, help='weighting factor for KL loss')
parser.add_argument('--emb_dim', type=int, default=4,
                    help='dimensionality of conserved embedding')
parser.add_argument('--last_frame_skip', action='store_true',
                    help='skip connection config')
parser.add_argument('--num_trials', type=int, default=5,
                    help='how many times to run training procedure')
parser.add_argument('--inner_crit_mode', default='mse',
                    help='"mse" or "cosine"')
parser.add_argument('--enc_dec_type', default='basic',
                    help='"basic" or "less_basic" or "vgg"')
parser.add_argument('--emb_type', default='basic',
                    help='"basic" or "conserved"')
parser.add_argument('--random_weights', action='store_true',
                    help='randomly init SVG weights?')
parser.add_argument('--outer_opt_model_weights', action='store_true',
                    help='optimize SVG weights in outer loop?')
parser.add_argument('--learn_inner_lr', action='store_true',
                    help='optimize inner LR in outer loop?')
parser.add_argument('--reuse_lstm_eps', action='store_true',
                    help='correlated eps samples for prior & posterior?')
parser.add_argument('--only_tailor_on_improvement', action='store_true',
                    help='no outer update if no inner improvement')
parser.add_argument('--only_cn_decoder', action='store_true',
                    help='CN layers in just decoder or encoder as well?')
parser.add_argument('--stack_frames', action='store_true',
                    help='stack every 2 frames channel-wise')
parser.add_argument('--only_twenty_degree', action='store_true',
                    help='for Phys101 ramp, only 20 degree setting?')
parser.add_argument('--center_crop', type=int, default=1080,
                    help='center crop param (phys101)')
parser.add_argument('--crop_upper_right', type=int,
                    default=1080, help='upper right crop param (phys101)')
parser.add_argument('--frame_step', type=int, default=2,
                    help='controls frame rate for Phys101')
parser.add_argument('--num_emb_frames', type=int, default=2,
                    help='number of frames to pass to the embedding')
parser.add_argument('--horiz_flip', action='store_true',
                    help='randomly flip phys101 sequences horizontally (p=.5)?')
parser.add_argument('--train_set_length', type=int,
                    default=-1, help='size of training set')
parser.add_argument('--test_set_length', type=int,
                    default=-1, help='size of test set')
parser.add_argument('--baseline', action='store_true',
                    help='evaluate baseline as well?')
parser.add_argument('--stop_grad', action='store_true',
                    help='perform stop grad?')
parser.add_argument('--inner_crit_compare_to',
                    default='prev', help='zero or prev')
parser.add_argument('--encoder_emb', action='store_true',
                    help='use EncoderEmbedding or ConservedEmbedding?')
parser.add_argument('--optimize_emb_enc_params', action='store_true',
                    help='optimize emb.encoder as well as enc.linear?')
parser.add_argument('--z_dim', type=int, default=10,
                    help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128,
                    help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--a_dim', type=int, default=8,
                    help='dimensionality of action, or a_t')
parser.add_argument('--fno_modes', type=int, default=12,
                    help='Number of FNO modes to keep')
parser.add_argument('--fno_width', type=int, default=20,
                    help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int,
                    default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int,
                    default=1, help='number of layers')
parser.add_argument('--fno_layers', type=int,
                    default=4, help='number of layers')
parser.add_argument('--no_teacher_force', action='store_true',
                    help='whether or not to do teacher forcing')
parser.add_argument('--add_inner_to_outer_loss', action='store_true',
                    help='optimize inner loss term in outer loop?')
parser.add_argument('--inner_opt_all_model_weights', action='store_true',
                    help='optimize non-CN model weights in inner loop?')
parser.add_argument('--batch_norm_to_group_norm',
                    action='store_true', help='replace BN layers with GN layers')
parser.add_argument('--conv_emb', action='store_true',
                    help='use fully-convolutional embedding?')
parser.add_argument('--pde_emb', action='store_true',
                    help='use PDE embedding?')
parser.add_argument('--pde_const_emb', action='store_true',
                    help='PDE embedding without learning parameters.')
parser.add_argument('--verbose', action='store_true', help='print loss info')
parser.add_argument('--warmstart_emb_path', default='',
                    help='path to pretrained embedding weights')

parser.add_argument('--param_loss', action = 'store_true',
                    help='parameter supervision to pre-train embedding model')
parser.add_argument('--num_learned_parameters', type=int, default=3,
                    help='number of parameters to learn in PDE embedding')
parser.add_argument('--num_param_combinations', type=int, default=-1,
                    help='number of parameters combinations to use in dataloader')
parser.add_argument('--fixed_ic', action = 'store_true',
                    help='train on a single initial condition for each parameter combination')
parser.add_argument('--fixed_window', action = 'store_true',
                    help='train on a single window of the trajectory for each parameter/IC combination')
parser.add_argument('--use_partials', action = 'store_true',
                    help='input partial derivatives into embedding model in addition to solution field')
parser.add_argument('--reload_best', action='store_true', help='reload best model')
parser.add_argument('--reload_checkpoint', action='store_true', help='reload latest model')
parser.add_argument('--save_checkpoint', action='store_true', help='checkpoint model')
parser.add_argument('--burgers_emb', action='store_true', help='use burgers embedding')
parser.add_argument('--advection_emb', action='store_true', help='use advection embedding')
parser.add_argument('--diffusion_reaction_emb', action='store_true', help='use 1d diffusion reaction  embedding')
# parser.add_argument('--train_embedding', default=True, type=bool, help='dummy flag indicating we are training the embedding model only. DO NOT CHANGE')


opt = parser.parse_args()
os.makedirs('%s' % opt.log_dir, exist_ok=True)

opt.n_eval = opt.n_past+opt.n_future
opt.max_step = opt.n_eval

if opt.image_width == 64:
    import models.vgg_64 as model
elif opt.image_width == 128:
    import models.vgg_128 as model
else:
    raise ValueError('image width must be 64 or 128')

val_inner_lr = opt.inner_lr
if opt.val_inner_lr != -1:
    val_inner_lr = opt.val_inner_lr

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# --------- tensorboard configs -------------------------------
tailor_str = 'None'
if opt.tailor:
    if opt.pde_emb:
        tailor_str = 'PDE'
    elif opt.conv_emb:
        tailor_str = 'Conv'
    elif opt.pde_const_emb:
        tailor_str = 'PDE_Const'

save_dir = os.path.join(opt.log_dir,
                                    str(datetime.now().ctime().replace(' ', '-').replace(':', '.')) +
                                    f'_past={opt.n_past}_future={opt.n_future}_tailor={tailor_str}')
writer = SummaryWriter(save_dir)
#dump params to yml
dump_params_to_yml(opt, save_dir)

if opt.tailor:
    max_tailor_steps = opt.num_inner_steps + 1
    custom_scalars = {
        "Inner Loss": {
            "Train": ["Multiline", [f"Inner Loss/train/{i} Steps" for i in range(max_tailor_steps)]],
            "Val": ["Multiline", [f"Inner Loss/val/{i} Steps" for i in range(max_tailor_steps)]],
        },
        "SVG Loss": {
            "Train": ["Multiline", [f"SVG Loss/train/{i} Steps" for i in range(max_tailor_steps)]],
            "Val": ["Multiline", [f"SVG Loss/val/{i} Steps" for i in range(max_tailor_steps)]],
        },
    }
    writer.add_custom_scalars(custom_scalars)


# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

if opt.stack_frames:
    assert opt.n_past % 2 == 0 and opt.n_future % 2 == 0
    opt.channels *= 2
    opt.n_past = opt.n_past // 2
    opt.n_future = opt.n_future // 2
opt.n_eval = opt.n_past+opt.n_future  # this is the sequence length
opt.max_step = opt.n_eval

if (opt.num_train_batch == -1) or (len(train_data) // opt.batch_size < opt.num_train_batch):
    opt.num_train_batch = len(train_data) // opt.batch_size
if (opt.num_val_batch == -1) or (len(test_data) // opt.batch_size < opt.num_val_batch):
    opt.num_val_batch = len(test_data) // opt.batch_size

train_loader = DataLoader(train_data,
                          num_workers=opt.num_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=False)
test_loader = DataLoader(test_data,
                         num_workers=opt.num_threads,
                         batch_size=opt.batch_size,
                         shuffle=False,
                         drop_last=True,
                         pin_memory=False)


# def get_batch_generator(data_loader):
#     while True:
#         for (data, params) in data_loader:
#             batch = torch.stack(utils.normalize_data(opt, dtype, data), dim =1)
#             yield batch, params


def get_batch_generator(data_loader):
    while True:
        for (data, params) in data_loader:
            batch = utils.normalize_data(opt, dtype, data)
            yield batch, params


training_batch_generator = get_batch_generator(train_loader)
testing_batch_generator = get_batch_generator(test_loader)


print(opt)
print('\nDatasets loaded!')

print(f'train_data length: {len(train_data)}')
print(f'num_train_batch: {opt.num_train_batch}')
print(f'test_data length: {len(test_data)}')
print(f'num_val_batch: {opt.num_val_batch}')




# --------- tracking metrics ------------------------------------
all_outer_losses = []
all_inner_losses = []
all_baseline_outer_losses = []
all_val_outer_losses = []
all_val_inner_losses = []
train_fstate_dict = []  # We'll write higher's fmodel.state_dict() here to compare
val_fstate_dict = []  # We'll write higher's fmodel.state_dict() here to compare
all_emb_weights = []
all_emb_biases = []
all_param_grads = []
all_grad_norms = []
all_emb_norms = []
all_inner_lr_scales = []


# --------- meta-training loop ------------------------------------
# usually only do one trial -- a trial is basically a run through the
# entire meta-training loop      
if opt.dataset not in  set(['1d_burgers_multiparam','1d_advection_multiparam','1d_diffusion_reaction_multiparam']):
                    print("2D EMBEDDING TRUE")
                    true_pde_embedding = TwoDDiffusionReactionEmbedding(in_size=opt.image_width,
                                                                in_channels=opt.channels, n_frames=opt.num_emb_frames, hidden_channels=opt.fno_width,
                                                                n_layers=opt.fno_layers, data_root=opt.data_root, learned=False)
else:
    true_pde_embedding = OneDEmbedding(in_size = opt.image_width ,
                                        in_channels =opt.channels, 
                                        n_frames=opt.num_emb_frames, 
                                        hidden_channels=opt.fno_width,
                                        n_layers=opt.fno_layers, 
                                        pde = opt.dataset,
                                        data_root=opt.data_root, learned=False)
    
if not opt.tailor: #if we're not tailoring, then make the embedding the non-learnable (i.e "true") PDE residual so we can log the inner loss
    embedding = true_pde_embedding
    same = True
    print('No tailoring - initialized Constant PDE ConservedEmbedding for inner loss logging')

elif opt.emb_type == 'conv_emb':
    embedding = ConvConservedEmbedding(image_width=opt.image_width,
                                        nc=opt.num_emb_frames * opt.channels)
    print('initialized Convolutional ConservedEmbedding')
elif opt.emb_type == 'pde_emb':

    if opt.dataset not in  set(['1d_burgers_multiparam','1d_advection_multiparam','1d_diffusion_reaction_multiparam']):
        embedding = TwoDDiffusionReactionEmbedding(in_size=opt.image_width,
                                                    in_channels=opt.channels, n_frames=opt.num_emb_frames, hidden_channels=opt.fno_width,
                                                    n_layers=opt.fno_layers, data_root=opt.data_root, learned=True, num_learned_parameters = opt.num_learned_parameters, use_partials = opt.use_partials)
    else:
        embedding = OneDEmbedding(in_size = opt.image_width ,
                                    in_channels =opt.channels, 
                                    n_frames=opt.num_emb_frames, 
                                    hidden_channels=opt.fno_width,
                                    n_layers=opt.fno_layers, 
                                    pde = opt.dataset,
                                    data_root=opt.data_root, learned=True,
                                    num_learned_parameters = opt.num_learned_parameters,
                                    use_partials = opt.use_partials)

    print('initialized Learnable PDE ConservedEmbedding')

elif opt.emb_type == 'pde_const_emb':
    # pdb.set_trace()
    embedding = true_pde_embedding
    same = True
    print('initialized Constant PDE ConservedEmbedding')
else:
    # embedding model
    # pdb.set_trace() 
    embedding = ConservedEmbedding(emb_dim=opt.emb_dim, image_width=opt.image_width,
                                    nc=opt.num_emb_frames * opt.channels)

    print('initialized ConservedEmbedding')

# In the case where we don't do tailoring, we can drop the embedding
# embedding = nn.Identity() if not opt.tailor else embedding

# print('emb summary')
# summary(embedding, input_size=(1, opt.num_emb_frames * opt.channels, opt.image_width, opt.image_width), dtypes=[torch.float64], device=torch.device("cuda"))
frame_predictor = FNO1d(n_modes_height=opt.fno_modes, hidden_channels=opt.fno_width,
                                in_channels=opt.channels*opt.n_past, out_channels=opt.channels, n_layers=opt.fno_layers)

# Init optimizer'=
# embedding = embedding.apply(lambda t: t.double())
embedding = embedding.to(torch.device("cuda")).to(torch.float64)#.double()
# frame_predictor = frame_predictor.apply(lt.to(torch.device("cuda")).to(torch.float64)#.double()
# params = [p[1] for p in embedding.named_parameters() if not (
    # 'gamma' in p[0] or 'beta' in p[0])]

pdb.set_trace()
# define outer optimizer
optimizer = optim.Adam([p for p in frame_predictor.parameters()]+ [p for p in embedding.parameters()], lr=opt.outer_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=15)

baseline_outer_losses = []
outer_losses = []
svg_losses = []
val_svg_losses = []
inner_losses = []
val_outer_losses = []
val_inner_losses = []
emb_weights = []
emb_biases = []
all_gen = None
param_grads = []
grad_norms = []
emb_norms = []

# Quick sanity check to ensure float64
embedding = embedding.apply(lambda t: t.to(torch.float64))
# embedding = embedding.double() .
for p in embedding.named_parameters():
    assert p[
        1].dtype == torch.float64, f'One of the embedding Model parameters is not float64! Parameter: {p[1], print(p)}'

print(f'starting at epoch 0')
train_losses = []
train_true_losses = []
val_losses = []
val_true_losses = []

train_nu_losses = []
train_rho_losses = []
train_nu_means = []
train_nu_vars = []
train_param_losses = []

val_nu_losses = []
val_nu_means = []
val_nu_vars = []
val_param_losses = []
val_rho_losses = []
val_loss_min_tracker = float("inf")


def save_checkpoint(embedding, log_dir, best = False):
    if best == False:
        torch.save({'model_state': embedding.state_dict()},'%s/ckpt_model.pt' % (log_dir))
    else:
        torch.save({'model_state': embedding.state_dict()},'%s/best_ckpt_model.pt' % (log_dir))


def restore_checkpoint(model, log_dir, device, best = False):
    if len(os.listdir(log_dir)) == 2:
        chosen_file = 'best_ckpt_model.pt' if best else 'ckpt_model.pt'
        checkpoint_path = os.path.join(log_dir, chosen_file)
        checkpoint = torch.load(checkpoint_path, map_location= device)
        model.load_state_dict(checkpoint['model_state'])




if opt.reload_checkpoint:
    restore_checkpoint(embedding, save_dir, torch.device("cuda") , opt.reload_best)

import pdb
# param_df = pd.DataFrame(columns = ['trueparam {}'.format(i) for i in range(opt.batch_size)] + ['outputparam {}'.format(i) for i in range(opt.batch_size)] + ['batch', 'epoch','train'])
for epoch in range(0, opt.n_epochs):

    print(f'Epoch {epoch} of {opt.n_epochs}')
    
    embedding.eval()

    # validation
    if epoch % opt.num_epochs_per_val == 0:
        print(f'Validation {epoch} Epoch')
        val_loss = 0
        val_true_loss = 0
        val_nu_loss = 0
        val_param_loss = 0
        val_nu_mean =  0
        val_nu_var =  0
        val_rho_loss = 0

        
        with torch.no_grad():
            for batch_num in tqdm(range(opt.num_val_batch)):
                batch, params = next(testing_batch_generator)
                pdb.set_trace()
                params = tuple([param.to(torch.device("cuda")).to(torch.float64) for param in params])
                batch = [data.to(torch.device("cuda")).to(torch.float64) for data in batch]
                stacked_batch = torch.stack(batch, dim =1)
                pde_value, true_pde_value, pred_params = embedding(stacked_batch, return_params = True, true_params = params)
                # print("(true_pde_value == 0).sum()",(true_pde_value == 0).sum())
                val_loss += torch.abs(pde_value+ 1e-10).log10().mean()
                val_true_loss += torch.abs(true_pde_value+ 1e-10).log10().mean()
                nu_pred = pred_params[0]
                nu = params[0]
                nu = nu.to(torch.device("cuda"))
                val_param_loss += (nu_pred - nu).pow(2).mean() #+ (rho_pred - rho).pow(2).mean()
                val_nu_loss += ((nu_pred - nu).abs() / nu).mean()

                phi_hat = torch.tile(pred_params[0][:,None,None,None], dims = (1,1,1024,1))
                # batch = torch.cat((stacked_batch, phi_hat),axis = 1)
                batch = [torch.cat((data, phi_hat),axis = 1) for data in batch]
                gen_seq, mus, logvars, mu_ps, logvar_ps = predict_many_steps_non_meta(frame_predictor, 
                                                                                        batch, 
                                                                                        None, 
                                                                                        opt,
                                                                                        mode='eval', 
                                                                                        prior_epses=[], 
                                                                                        posterior_epses=[], 
                                                                                        learnable_model = None)

                if opt.learned_pinn_loss and opt.pinn_outer_loss:
                    outer_mse_loss, outer_pde_loss = svg_crit(
                        gen_seq, batch, mus, logvars, mu_ps, logvar_ps, embedding, params, opt)

                else:
                    outer_mse_loss, outer_pde_loss = svg_crit(
                        gen_seq, batch, mus, logvars, mu_ps, logvar_ps, true_pde_embedding, params, opt)
                    outer_mse_loss = outer_mse_loss.mean()
                    outer_pde_loss = outer_pde_loss.mean()
                if opt.no_data_loss and opt.pinn_outer_loss:
                    outer_loss = outer_pde_loss #total data + PDE loss                
                else:
                    outer_loss = outer_mse_loss + outer_pde_loss

                param_loss = (nu_pred - nu).pow(2).mean() #+ (rho_pred - rho).pow(2).mean()
                if opt.param_loss:
                    embedding_loss = param_loss#.backward()
                else:
                    embedding_loss = val_loss#.backward()

                total_loss = embedding_loss + outer_loss


            #step scheduler
            if opt.param_loss:
                scheduler.step(val_param_loss)
            else:
                scheduler.step(val_loss)


            if opt.save_checkpoint:
                if opt.param_loss:
                    if  val_loss_min_tracker > val_param_loss / opt.num_val_batch:
                        val_loss_min_tracker = val_param_loss / opt.num_val_batch
                        save_checkpoint(embedding,save_dir, True)
                else:
                    if  val_loss_min_tracker > val_loss / opt.num_val_batch:
                        val_loss_min_tracker = val_loss / opt.num_val_batch
                        save_checkpoint(embedding,save_dir, True)


            val_losses.append(val_loss / opt.num_val_batch)
            val_true_losses.append(val_true_loss / opt.num_val_batch)
            val_nu_losses.append(val_nu_loss / opt.num_val_batch)
            val_nu_means.append(val_nu_mean / opt.num_val_batch)
            val_nu_vars.append(val_nu_var / opt.num_val_batch)
            val_param_losses.append(val_param_loss / opt.num_val_batch )
            val_rho_losses.append(val_rho_loss / opt.num_val_batch)



    print("Val PDE Loss: ", val_loss / opt.num_val_batch)
    embedding.train()
    # Training
    print(f'Train {epoch} Epoch')
    train_loss = 0
    train_true_loss = 0
    train_nu_loss = 0
    train_rho_loss = 0
    train_param_loss = 0

    train_nu_mean = 0

    train_nu_var = 0
    

    for batch_num in tqdm(range(opt.num_train_batch)):
        optimizer.zero_grad()
        batch, params = next(training_batch_generator)
        # data = data.reshape(-1, data.shape[-2], data.shape[-1])
        # params = torch.repeat_interleave(params[0],4)
        # pdb.set_trace()
        params = tuple([param.to(torch.device("cuda")).to(torch.float64) for param in params])
        stacked_batch = torch.stack(batch, dim =1)
        pde_value, true_pde_value, pred_params = embedding(stacked_batch, return_params = True, true_params = params)
        loss = (pde_value+ 1e-10 ).abs().log10().mean()#.detach()
        loss = loss.detach() if opt.param_loss else loss
        # print("(true_pde_value == 0).sum()",(true_pde_value == 0).sum())
        true_loss = (true_pde_value+ 1e-10).abs().log10().mean()
        # print("val_pred_params",pred_params)        
        train_loss  += loss
        train_true_loss += true_loss
        nu_pred = pred_params[0]
        nu = params[0]
        nu = nu.to(torch.device("cuda"))
        train_nu_loss += ((nu_pred - nu).abs() / nu).mean()
        
        phi_hat = torch.tile(pred_params[0][:,None,None,None,None], dims = (1,1,1,1024,1))
        batch = torch.cat((batch, phi_hat),axis = 1)

        gen_seq, mus, logvars, mu_ps, logvar_ps = predict_many_steps_non_meta(frame_predictor, 
                                                                    batch.to(torch.float64), 
                                                                    None, 
                                                                    opt,
                                                                    mode='eval', 
                                                                    prior_epses=[], 
                                                                    posterior_epses=[], 
                                                                    learnable_model = False)

        if opt.learned_pinn_loss and opt.pinn_outer_loss:
            outer_mse_loss, outer_pde_loss = svg_crit(
                gen_seq, batch, mus, logvars, mu_ps, logvar_ps, embedding, params, opt)

        else:
            outer_mse_loss, outer_pde_loss = svg_crit(
                gen_seq, batch, mus, logvars, mu_ps, logvar_ps, true_pde_embedding, params, opt)
            outer_mse_loss = outer_mse_loss.mean()
            outer_pde_loss = outer_pde_loss.mean()
        if opt.no_data_loss and opt.pinn_outer_loss:
            outer_loss = outer_pde_loss #total data + PDE loss                
        else:
            outer_loss = outer_mse_loss + outer_pde_loss

        param_loss = (nu_pred - nu).pow(2).mean() #+ (rho_pred - rho).pow(2).mean()
        if opt.param_loss:
            embedding_loss = param_loss#.backward()
        else:
            embedding_loss = loss#.backward()

        total_loss = embedding_loss + outer_loss
        total_loss.backward()
        #gradient clipping
        torch.nn.utils.clip_grad_norm_(embedding.parameters(), max_norm = 1)
        optimizer.step()
        train_param_loss+=param_loss

        train_nu_mean += nu.mean()
        train_nu_var += nu.var()

        # param_df.loc[len(param_df),param_df.columns] = params[0].tolist() + pred_params[0].tolist() + [batch_num, epoch, 1]

    if opt.save_checkpoint:
        save_checkpoint(embedding,save_dir, False)
    

    train_losses.append(train_loss / opt.num_train_batch)
    train_true_losses.append(train_true_loss / opt.num_train_batch)
    train_nu_losses.append(train_nu_loss / opt.num_train_batch)
    train_rho_losses.append(train_rho_loss / opt.num_train_batch)
    train_nu_means.append(train_nu_mean / opt.num_train_batch)
    train_nu_vars.append(train_nu_var / opt.num_train_batch)
    train_param_losses.append(train_param_loss / opt.num_train_batch)
    print("Train PDE Loss: ", train_loss / opt.num_train_batch)
    
    # #write to tensorboard
    writer.add_scalar('val_log_pde_loss', val_losses[-1],(epoch + 1))
    writer.add_scalar('val_log_true_pde_loss', val_true_losses[-1],(epoch + 1))
    writer.add_scalar('val_nu_loss', val_nu_losses[-1],(epoch + 1))
    writer.add_scalar('val_rho_loss', val_rho_losses[-1],(epoch + 1))
    writer.add_scalar('val_param_loss', val_param_losses[-1],(epoch + 1))



    writer.add_scalar('val_nu_mean', val_nu_means[-1],(epoch + 1))
    writer.add_scalar('val_nu_var', val_nu_vars[-1],(epoch + 1))


    writer.add_scalar('train_log_pde_loss', train_losses[-1],(epoch + 1))
    writer.add_scalar('train_log_true_pde_loss', train_true_losses[-1],(epoch + 1))
    writer.add_scalar('train_nu_loss', train_nu_losses[-1],(epoch + 1))
    writer.add_scalar('train_rho_loss', train_rho_losses[-1],(epoch + 1))
    writer.add_scalar('train_param_loss', train_param_losses[-1],(epoch + 1))


    writer.add_scalar('train_nu_mean', train_nu_means[-1],(epoch + 1))
    writer.add_scalar('train_nu_var', train_nu_vars[-1],(epoch + 1))

# param_df.to_csv('param_df.csv')
hyperparameters = {
    'tailor': tailor_str,
    'batch_size': opt.batch_size,
    'seed': opt.seed,
    'n_past': opt.n_past,
    'n_future': opt.n_future,
    'dataset': opt.dataset,
    'image_width': opt.image_width,
    'n_epochs': opt.n_epochs,
    'channels': opt.channels,
    'num_epochs_per_val': opt.num_epochs_per_val,
    'num_inner_steps': opt.num_inner_steps,
    'num_jump_steps': opt.num_jump_steps,
    'num_train_batch': opt.num_train_batch,
    'num_val_batch': opt.num_val_batch,
    'inner_lr': opt.inner_lr,
    'val_inner_lr': opt.val_inner_lr,
    'outer_lr': opt.outer_lr,
    'emb_dim': opt.emb_dim,
    'num_trials': opt.num_trials,
    'inner_crit_mode': opt.inner_crit_mode,
    'enc_dec_type': opt.enc_dec_type,
    'emb_type': opt.emb_type,
    'outer_opt_model_weights': opt.outer_opt_model_weights,
    'reuse_lstm_eps': opt.reuse_lstm_eps,
    'only_twenty_degree': opt.only_twenty_degree,
    'center_crop': opt.center_crop,
    'crop_upper_right': opt.crop_upper_right,
    'frame_step': opt.frame_step,
    'num_emb_frames': opt.num_emb_frames,
    'horiz_flip': opt.horiz_flip,
    'train_set_length': opt.train_set_length,
    'test_set_length': opt.test_set_length,
    'encoder_emb': opt.encoder_emb,
    'z_dim': opt.z_dim,
    'g_dim': opt.g_dim,
    'a_dim': opt.a_dim,
    'fno_modes': opt.fno_modes,
    'fno_width': opt.fno_width,
    'fno_layers': opt.fno_layers,
    'inner_opt_all_model_weights': opt.inner_opt_all_model_weights,
    'batch_norm_to_group_norm': opt.batch_norm_to_group_norm,
    'conv_emb': opt.conv_emb,
    'pde_emb': opt.pde_emb,
    'pde_const_emb': opt.pde_const_emb,
}
hparams_logging = hparams(hyperparameters, {})
for i in hparams_logging:
    writer.file_writer.add_summary(i)

writer.flush()
writer.close()

import torch
# This sets the default model weights to float64
# torch.set_default_dtype(torch.float64)  # nopep8
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
from torch.utils.data import DataLoader
import utils
from utils import svg_crit, dump_params_to_yml, compute_losses,plot_solution_field
import itertools
import numpy as np
import copy
import higher
from datetime import datetime
from torch.utils.tensorboard.summary import hparams
import pdb
from models.forward import dont_tailor_many_steps#, tailor_many_steps, pino_style_tailoring
from models.cn import replace_cn_layers
from models.svg import SVGModel
from models.fno_models import FNOEncoder, FNODecoder
from torch.nn import Linear, Conv2d, ReLU
from models.embedding import ConservedEmbedding, ConvConservedEmbedding, TwoDDiffusionReactionEmbedding
from models.OneD_embeddings import OneDEmbedding
import models.lstm as lstm_models
import matplotlib.pyplot as plt
from neuraloperator.neuralop.models import FNO, FNO1d, CNFNO1d, CNFNO
import copy
from torchinfo import summary
import pickle
# NOTE: deterministic for debugging
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--train_noether', default=True, type=bool, help='dummy flag indicating we are training the joint noether model. DO NOT CHANGE')
parser.add_argument('--train_batch_size', default=100, type=int, help='batch size')
parser.add_argument('--val_batch_size', default=1, type=int, help='batch size')
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
parser.add_argument('--pinn_outer_loss', action='store_true',
                    help='if true, include the (true) PDE residual in outer loss')
parser.add_argument('--pinn_outer_loss_weight', type=float,
                    default=1.0, help='weight for PDE residual in outer loss')
parser.add_argument('--train_num_inner_steps', type=int,
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
                    help='type of embedding - pde, pde_conv, pde_const, or naive conserved')
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
parser.add_argument('--verbose', action='store_true', help='print loss info')
parser.add_argument('--warmstart_emb_path', default='',
                    help='path to pretrained embedding weights')
parser.add_argument('--use_partials', action = 'store_true',
                    help='input partial derivatives into embedding model in addition to solution field')
parser.add_argument('--num_learned_parameters', type=int, default=3,
                    help='number of parameters to learn in PDE embedding')
parser.add_argument('--save_checkpoint', action='store_true', 
                    help='to checkpoint models')
parser.add_argument('--ckpt_outer_loss', action='store_true',
                    help='to checkpoint best validation outer loss')
parser.add_argument('--ckpt_inner_loss', action='store_true', 
                    help='to checkpoint best validation inner loss')
parser.add_argument('--ckpt_svg_loss', action='store_true', 
                    help='to checkpoint best validation svg loss')

parser.add_argument('--reload_latest', action='store_true', 
                    help='to reload last checkpoint model')
parser.add_argument('--reload_best_outer', action='store_true',
                    help='to reload best outer val. model')
parser.add_argument('--reload_best_inner', action='store_true', 
                    help='to reload best inner val. model')
parser.add_argument('--reload_best_svg', action='store_true', 
                    help='to reload best svg val. model')
parser.add_argument('--num_param_combinations', type=int,default=-1)
parser.add_argument('--fixed_ic', action='store_true')
parser.add_argument('--fixed_window', action='store_true')
parser.add_argument('--learned_pinn_loss', action = 'store_true')
parser.add_argument('--no_data_loss',action = 'store_true')
parser.add_argument('--add_general_learnable', action='store_true',
                    help='use FNO in tailoring step')
parser.add_argument('--param_loss', action='store_true',
                    help='use param_loss for embedding network')
parser.add_argument('--use_embedding', action='store_true',
                    help='use the loss from g network in backprop')
parser.add_argument('--teacher_forcing', action='store_true',
                    help='use teacher forcing')
parser.add_argument('--use_true_params_val', action = 'store_true',
                    help='use true params in dynamics model during validation')
parser.add_argument('--use_true_params_train', action = 'store_true',
                    help ='use predicted params in dynamics model during training')
parser.add_argument('--use_cn', action = 'store_true',
                    help ='use conditional normalization')
parser.add_argument('--conditioning', action = 'store_true',
                    help ='to condition on parameters')
parser.add_argument('--outer_loss_choice', type = str, default = 'original',
                    help ='pde loss choice')
parser.add_argument('--operator_loss', action = 'store_true',
                        help = 'use operator loss per instance optimization')
parser.add_argument('--num_tuning_steps', type = int, default = 1,
                        help = 'number of times to tune at test time ')
parser.add_argument('--relative_data_loss', action = 'store_true',
                    help = 'take relative data loss for optimization')
parser.add_argument('--mean_reduction', action = 'store_true',
                    help = 'take mean of inner losses')
parser.add_argument('--use_adam_inner_opt', action = 'store_true',
                    help ='to use adam as inner optimizer')
parser.add_argument('--norm', default = '',
                    help ='to use select between instance_norm, layer_norm, batch_norm')
parser.add_argument('--reload_dir', type=str, help='reload ckpt file path')
parser.add_argument('--ood', action = 'store_true',
                    help='ood exp for advection')

print("torch.cuda.current_device()",torch.cuda.current_device())
device = torch.device('cuda')
opt = parser.parse_args()
os.makedirs('%s' % opt.log_dir, exist_ok=True)

opt.n_eval = opt.n_past+opt.n_future if opt.n_future != -1 else -1
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
np.random.seed(opt.seed)
# dtype = torch.cuda.DoubleTensor
dtype = torch.cuda.FloatTensor

# --------- tensorboard configs -------------------------------
tailor_str = 'None'
if opt.tailor:
    if opt.emb_type == 'pde_emb':
        tailor_str = 'PDE'
    elif opt.emb_type == 'conv_emb':
        tailor_str = 'Conv'
    elif opt.emb_type == "pde_const_emb":
        tailor_str = 'PDE_Const'
    tailor_str += f'_{opt.train_num_inner_steps}'
if opt.pinn_outer_loss:
    tailor_str+=f'_PINN_Outer_Loss_Weight={opt.pinn_outer_loss_weight}'

save_dir = os.path.join(opt.log_dir,
                                    str(datetime.now().ctime().replace(' ', '-').replace(':', '.')) +
                                    f'_past={opt.n_past}_future={opt.n_future}_tailor={tailor_str}')
writer = SummaryWriter(save_dir)
#dump params to yml
dump_params_to_yml(opt, save_dir)

#if opt.tailor:
#want to measure PDE residual loss even when not tailoring
max_tailor_steps = opt.train_num_inner_steps + 1
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

if (opt.num_train_batch == -1) or (len(train_data) // opt.train_batch_size < opt.num_train_batch):
    opt.num_train_batch = len(train_data) // opt.train_batch_size
if (opt.num_val_batch == -1) or (len(test_data) // opt.val_batch_size < opt.num_val_batch):
    opt.num_val_batch = len(test_data) // opt.val_batch_size

train_loader = DataLoader(train_data,
                          num_workers=opt.num_threads,
                          batch_size=opt.train_batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=False)
test_loader = DataLoader(test_data,
                        num_workers=opt.num_threads,
                        batch_size=opt.val_batch_size,
                        shuffle=False,
                        drop_last=True,
                        pin_memory=False)


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
# import pdb
#  

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

learnable_model = None


def save_checkpoint(model, log_dir, best_outer = False, best_inner = False, best_svg = False):
    chosen_file = ''
    if best_outer == False and best_inner == False and best_svg == False:
        chosen_file = 'ckpt_model.pt'
    elif best_outer == True and best_inner == False and best_svg == False:
        chosen_file = 'best_outer_val_ckpt_model.pt'
    elif best_outer == False and best_inner == True and best_svg == False:
        chosen_file = 'best_inner_val_ckpt_model.pt'
    elif best_outer == False and best_inner == False and best_svg == True:
        chosen_file = 'best_svg_val_ckpt_model.pt'
    torch.save({'model_state': model.state_dict()},'%s/%s' % (log_dir, chosen_file))


def restore_checkpoint(model, log_dir, device, best_outer = False, best_inner = False, best_svg = False):
    if len(os.listdir(log_dir)) >= 2:
        
        chosen_file = ''
        if best_outer == False and best_inner == False and best_svg == False:
            chosen_file = 'ckpt_model.pt'
        elif best_outer == True and best_inner == False and best_svg == False:
            chosen_file = 'best_outer_val_ckpt_model.pt'
        elif best_outer == False and best_inner == True and best_svg == False:
            chosen_file = 'best_inner_val_ckpt_model.pt'
        elif best_outer == False and best_inner == False and best_svg == True:
            chosen_file = 'best_svg_val_ckpt_model.pt'
        
        checkpoint_path = os.path.join(log_dir, chosen_file)
        checkpoint = torch.load(checkpoint_path, map_location= device)
        model.load_state_dict(checkpoint['model_state'])

# save_dir = os.path.join(opt.log_dir,
#                                     str(datetime.now().ctime().replace(' ', '-').replace(':', '.')) +
#                                     f'_past={opt.n_past}_future={opt.n_future}_tailor={tailor_str}')
# os.makedirs(save_dir, exist_ok=True)

# --------- meta-training loop ------------------------------------
# usually only do one trial -- a trial is basically a run through the
# entire meta-training loop

for trial_num in range(opt.num_trials):
    start_epoch = 0

    # import pdb
    # 

    print(f'TRIAL {trial_num}')
    if opt.random_weights:
        print('initializing model with random weights')
        opt.a_dim = 0 if not opt.use_action else opt.a_dim
        # dynamics model
        # FNO1d(n_modes_height = 16, hidden_channels = hidden_channels,
        #     in_channels = in_channels, out_channels = hidden_channels, n_layers = n_layers).to(torch.complex128)
        if opt.dataset in  set(['1d_burgers_multiparam','1d_advection_multiparam','1d_diffusion_reaction_multiparam']):
            frame_predictor = CNFNO1d(n_modes_height=opt.fno_modes, 
                                    hidden_channels=opt.fno_width,
                                    in_channels=opt.channels*opt.n_past, 
                                    out_channels=opt.channels, 
                                    norm = opt.norm if opt.norm != '' else None,
                                    n_layers=opt.fno_layers,
                                    use_cn = opt.use_cn,
                                    val_batch_size = opt.val_batch_size)
            if opt.add_general_learnable == True:
            # print("h1")
                class inner_learnable(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fno = FNO1d(n_modes_height=opt.fno_modes, hidden_channels=opt.fno_width,
                in_channels=opt.channels*opt.n_past, out_channels=opt.channels, n_layers=opt.fno_layers).to(torch.cuda.current_device())
                        self.inner = nn.Linear(1024,1)
                    def forward(self, x):
                        return self.inner(self.fno(x))

                learnable_model = inner_learnable() #FNO1d(n_modes_height=opt.fno_modes, hidden_channels=opt.fno_width,
                # in_channels=opt.channels*opt.n_past, out_channels=opt.channels, n_layers=opt.fno_layers).to(torch.cuda.current_device())
                learnable_model.apply(lambda t: t.cuda())

        # frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim+opt.a_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.train_batch_size)
        else:
            print("2d fno")
            # frame_predictor = CNFNO(n_modes=(opt.fno_modes, opt.fno_modes), hidden_channels=opt.fno_width,
            #                     in_channels=opt.channels*opt.n_past, out_channels=opt.channels, n_layers=opt.fno_layers)
            frame_predictor = CNFNO(n_modes=(opt.fno_modes, opt.fno_modes), 
                                    hidden_channels=opt.fno_width,
                                    in_channels=opt.channels*opt.n_past, 
                                    out_channels=opt.channels, 
                                    n_layers=opt.fno_layers,
                                    val_batch_size = opt.val_batch_size,
                                    train_batch_size = opt.train_batch_size,
                                    use_cn = opt.use_cn,
                                    norm = opt.norm if opt.norm != '' else None,)

            if opt.add_general_learnable == True:
                 # print("h1")
                class inner_learnable(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fno = FNO(n_modes=(opt.fno_modes, opt.fno_modes), hidden_channels=opt.fno_width,
                                in_channels=opt.channels*opt.n_past, out_channels=opt.channels, n_layers=opt.fno_layers)

                        self.mlp = nn.Linear(32768, 1)
                    def forward(self, x):
                        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
                        x = self.fno(x)
                        x = torch.flatten(x, start_dim=1)
                        x = self.mlp(x)
                        return x
                learnable_model = inner_learnable() #FNO1d(n_modes_height=opt.fno_modes, hidden_channels=opt.fno_width,
                # in_channels=opt.channels*opt.n_past, out_channels=opt.channels, n_layers=opt.fno_layers).to(torch.cuda.current_device())
                learnable_model.apply(lambda t: t.cuda())

        posterior = nn.Identity()
        prior = nn.Identity()
        # posterior = lstm_models.gaussian_FNO(opt.g_dim+opt.a_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.train_batch_size)
        # prior = lstm_models.gaussian_FNO(opt.g_dim+opt.a_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.train_batch_size)
        # frame_predictor.apply(utils.init_weights)
        # frame_predictor.apply(utils.init_forget_bias_to_one)
        # posterior.apply(utils.init_weights)
        # prior.apply(utils.init_weights)

        encoder = nn.Identity()  # FNOEncoder()
        decoder = nn.Identity()  # FNOEncoder()


        # encoder = model.encoder(opt.g_dim, opt.channels, use_cn_layers=True, train_batch_size=opt.train_batch_size)
        # decoder = model.decoder(opt.g_dim, opt.channels, use_cn_layers=True, train_batch_size=opt.train_batch_size)
        # encoder.apply(utils.init_weights)
        # decoder.apply(utils.init_weights)

        same = False
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
            embedding = true_pde_embedding
            same = True
            print('initialized Constant PDE ConservedEmbedding')
        else:
            # embedding model
            embedding = ConservedEmbedding(emb_dim=opt.emb_dim, image_width=opt.image_width,
                                            nc=opt.num_emb_frames * opt.channels)

            print('initialized ConservedEmbedding')

        # In the case where we don't do tailoring, we can drop the embedding
        #EDIT: probably don't want to do this since we still want to compute the inner losses when we don't tailor
        #embedding = nn.Identity() if not opt.tailor else embedding

        # complete model
        svg_model = SVGModel(encoder, frame_predictor,
                             decoder, prior, posterior, embedding, true_pde_embedding).cuda()
        svg_model.apply(lambda t: t.cuda())

        if opt.reload_dir:
            checkpoint_path = os.path.join(opt.reload_dir, 'ckpt_model.pt') if 'ckpt_model.pt' not in opt.reload_dir else opt.reload_dir
            checkpoint = torch.load(checkpoint_path, map_location= device)
            svg_model.load_state_dict(checkpoint['model_state'])

    replace_cn_layers(svg_model.encoder)#, batch_size=opt.train_batch_size)
    replace_cn_layers(svg_model.decoder)#, batch_size=opt.train_batch_size)
    replace_cn_layers(svg_model.frame_predictor)
    svg_model.frame_predictor.batch_size = opt.train_batch_size
    svg_model.posterior.batch_size = opt.train_batch_size
    svg_model.prior.batch_size = opt.train_batch_size

    if opt.batch_norm_to_group_norm:
        print('replacing batch norm layers with group norm')
        svg_model = utils.batch_norm_to_group_norm(svg_model)

    #load pretrained embedding model
    if opt.warmstart_emb_path != '' and opt.tailor and opt.emb_type != "pde_const_emb":
        emb_ckpt = torch.load(opt.warmstart_emb_path)
        svg_model.emb.load_state_dict(emb_ckpt['model_state'])

    svg_model.apply(lambda t: t.cuda())
    # For comparing later
    # old_state_dict = copy.deepcopy(svg_model.state_dict())
    if opt.baseline:
        # only useful for tailoring or meta-tailoring from a checkpoint
        baseline_svg_model = copy.deepcopy(svg_model)
        for param in baseline_svg_model.parameters():
            param.requires_grad = False
        baseline_svg_model.cuda()
        baseline_svg_model.eval()

    # TODO NC: I'm pretty sure none of this needs to be changed since we're using identity now.

    # Init outer optimizer
    emb_params = [p[1] for p in svg_model.emb.named_parameters() if not (
        'gamma' in p[0] or 'beta' in p[0])]

    # Dont do this
    if opt.encoder_emb and not opt.optimize_emb_enc_params:
        emb_params = list(svg_model.emb.linear.parameters())

    # Do this
    if opt.outer_opt_model_weights:
        # optimize the non-CN model weights in the outer loop, as well as emb params

        # encoder and decoder
        non_cn_params = [p[1] for p in list(svg_model.encoder.named_parameters()) +
                         list(svg_model.decoder.named_parameters())
                         if not ('gamma' in p[0] or 'beta' in p[0])]
        outer_params = non_cn_params + emb_params + \
            list(svg_model.prior.parameters()) + \
            list(svg_model.posterior.parameters()) + \
            list(svg_model.frame_predictor.parameters())
    else:
        outer_params = emb_params

    # Don't do this
    if opt.learn_inner_lr:
        svg_model.inner_lr_scale = torch.nn.Parameter(torch.tensor(0.))
        outer_params.append(svg_model.inner_lr_scale)

    # define outer optimizer
    outer_opt = optim.Adam(outer_params, lr=opt.outer_lr)
    tailor_params = []
    frame_predictor_params =  list(svg_model.frame_predictor.named_parameters())
    if opt.use_cn == True and opt.inner_opt_all_model_weights == False:
        tailor_params += [p[1] for p in frame_predictor_params if ('gamma' in p[0] or 'beta' in p[0])]
    elif opt.use_cn == False and opt.inner_opt_all_model_weights == True:
        tailor_params += [p[1] for p in frame_predictor_params if not ('gamma' in p[0] or 'beta' in p[0])]
    elif opt.use_cn == True and opt.inner_opt_all_model_weights == True:
        tailor_params += list(svg_model.frame_predictor.parameters())

    if opt.use_adam_inner_opt:
        inner_opt = optim.Adam(tailor_params, lr=opt.inner_lr)
    else:
        inner_opt = optim.SGD(tailor_params, lr=opt.inner_lr)

    baseline_outer_losses = []
    outer_losses = []
    svg_losses = []
    val_svg_losses = []
    inner_losses = []
    true_inner_losses = []
    val_outer_losses = []
    val_pre_outer_losses = []
    val_relative_losses = []
    val_pre_relative_losses = []
    val_gain_outer_losses = []
    val_pre_inner_losses = []
    val_gain_inner_losses = []
    val_residual_losses = []
    val_pre_residual_losses = []
    val_gain_residual_losses = []
    val_inner_losses = []
    val_pre_log_inner_losses = []
    val_log_inner_losses = []
    val_true_inner_losses = []
    emb_weights = []
    emb_biases = []
    all_gen = None
    param_grads = []
    grad_norms = []
    emb_norms = []
    val_param_grads = []
    val_param_diffs = []
    param_losses = []
    true_param_losses = []
    val_param_losses = []
    val_true_param_losses = []
    
    inner_gain = []
    true_inner_gain = []
    val_inner_gain = []
    val_true_inner_gain = []

    min_val_outer_loss = float('inf')
    min_val_inner_loss = float('inf')
    min_val_svg_loss = float('inf')
    # Quick sanity check to ensure float64
    for p in svg_model.named_parameters():
        print(p[1].dtype)
        # assert p[
            # 1].dtype == torch.float64, f'One of the SVG Model parameters is not float64! Parameter: {p[1]}'
    print(f'starting at epoch {start_epoch}')
    val_pde_losses = []
    val_mse_losses = []
    val_rel_losses = []
    val_prediction_collector = []
    param_collector = []
    val_param_losses = []
    for epoch in range(1):

        print(f'Epoch {epoch} of {opt.n_epochs}')
        train_outer_loss = 0.
        grad_norm_sum = 0.
        emb_norm_sum = 0.
        epoch_inner_losses = []
        epoch_true_inner_losses = []
        epoch_val_inner_losses = []
        epoch_val_true_inner_losses = []

        epoch_param_losses = []
        epoch_true_param_losses = []
        epoch_val_param_losses = []
        epoch_val_true_param_losses = []

        epoch_inner_gain = []
        epoch_true_inner_gain = []
        epoch_val_inner_gain = []
        epoch_val_true_inner_gain = []

        epoch_svg_losses = []
        epoch_val_svg_losses = []
        svg_model.eval()

        # validation
        if epoch < 500:
            print(f'Validation {epoch} Epoch')
            val_outer_loss = 0.
            val_pre_outer_loss = 0.
            val_loss_diff = 0.
            val_pre_log_inner_loss = 0
            val_log_inner_loss = 0
            val_relative_loss = 0.
            val_pre_relative_loss = 0.

            val_post_residual = 0.
            val_pre_residual = 0.
            val_residual_diff = 0.

            val_inner_loss = 0.
            val_pre_inner_loss = 0.
            val_inner_loss_diff = 0.
            baseline_outer_loss = 0.

            loss_per_step_tracker = dict()
            rel_loss_per_step_tracker = dict()
            
            inner_loss_per_step_tracker = dict()
            rel_inner_loss_per_step_tracker = dict()
            tailor_param_grads = dict()
            grad_collector_all = []
            
            svg_model.eval()

            for batch_num in tqdm(range(opt.num_val_batch)):
                
                batch, params = next(testing_batch_generator)
                params = tuple([param.to(torch.cuda.current_device()) for param in params])
                # pde_value, true_pde_value, pred_params = embedding(data, return_params = True, true_params = params)
                with torch.no_grad():
                    # we optionally evaluate a baseline (untailored) model for comparison
                    prior_epses = []
                    posterior_epses = []

                    # Don't do this
                    if opt.baseline:
                        base_gen_seq, base_mus, base_logvars, base_mu_ps, base_logvar_ps = \
                            predict_many_steps(baseline_svg_model, batch, opt, mode='eval',
                                               prior_epses=prior_epses, posterior_epses=posterior_epses, learnable_model = learnable_model)
                        base_outer_mse_loss, base_outer_pde_loss,_ = svg_crit(base_gen_seq, batch, base_mus, base_logvars,
                                                   base_mu_ps, base_logvar_ps, true_pde_embedding, opt)
                        base_outer_mse_loss = base_outer_mse_loss.mean()
                        base_outer_pde_loss = base_outer_pde_loss.mean()
                        base_outer_loss = base_outer_mse_loss + base_outer_pde_loss
                # tailoring pass
                val_cached_cn = [None]  # cached cn params
                val_batch_inner_losses = []
                val_batch_true_inner_losses = []
                val_batch_svg_losses = []
                val_batch_inner_param_losses = []
                val_batch_true_inner_param_losses = []
                val_batch_inner_gain = []
                val_batch_true_inner_gain = []
                # pdb.set_trace()
                for batch_step in range(opt.num_jump_steps + 1):
                    # jump steps are effectively inner steps that have a single higher innerloop_ctx per
                    # iteration, which allows for many inner steps during training without running
                    # into memory issues due to storing the whole dynamic computational graph
                    # associated with unrolling the sequence in the inner loop for many steps
                    # 
                    pre_tailor_params = copy.deepcopy(svg_model.frame_predictor.state_dict())
                    with torch.no_grad():
                        init_gen_seq, mus, logvars, mu_ps, logvar_ps = dont_tailor_many_steps(
                            # no need for higher grads in val
                            svg_model, batch, true_pde_embedding, params, opt=opt, track_higher_grads=False,
                            mode='eval',learnable_model = learnable_model,
                        )
                        # outer_loss, outer_pde_loss, outer_mse_loss, pde_residual, outer_relative_loss
                        # post_opt_outer_loss, post_outer_pde_loss, post_outer_data_loss,post_pde_residual_avg, post_relative_loss, post_log_inner_loss = compute_losses(gen_seq, batch, mus, logvars, mu_ps, logvar_ps, true_pde_embedding, true_pde_embedding, params, opt)
                        opt_outer_loss, pre_outer_pde_loss, pre_outer_data_loss, pre_pde_residual_avg, pre_relative_loss, pre_log_inner_loss = compute_losses(init_gen_seq, batch, mus, logvars, mu_ps, logvar_ps, embedding, true_pde_embedding, params, opt)
                        val_pre_inner_loss += pre_outer_pde_loss.cpu().item()
                        val_pre_outer_loss += pre_outer_data_loss.detach().cpu().item() #only log the data loss
                        val_pre_relative_loss += pre_relative_loss
                        val_pre_residual += pre_pde_residual_avg.detach().cpu().item()
                        val_pre_log_inner_loss += pre_log_inner_loss.detach().cpu().item()
                    # perform tailoring (autoregressive prediction, tailoring, predict again)
                    # 
                    # Tailor many steps uses opt.tailor to decide whether to tailor or not
                    replace_cn_layers(svg_model.frame_predictor)
                    for tailor_step in range(opt.num_tuning_steps):
                        gen_seq, mus, logvars, mu_ps, logvar_ps = dont_tailor_many_steps(
                                                                    # no need for higher grads in val
                                                                    svg_model, batch, true_pde_embedding, params, opt=opt, track_higher_grads=False,
                                                                    mode='eval',learnable_model = learnable_model,param_collector = param_collector,
                                                                    # extra kwargs
                                                                )
                        tailor_opt_outer_loss, tailor_outer_pde_loss, tailor_outer_mse_loss,_,_,tailor_log_inner_loss = compute_losses(gen_seq, init_gen_seq, mus, logvars, mu_ps, logvar_ps, true_pde_embedding, true_pde_embedding, params, opt)
                        if tailor_step not in loss_per_step_tracker:
                            loss_per_step_tracker[tailor_step] = 0
                            inner_loss_per_step_tracker[tailor_step] = 0
                            tailor_param_grads[tailor_step] = 0
                        loss_per_step_tracker[tailor_step] += (tailor_outer_mse_loss).detach().cpu().item()
                        inner_loss_per_step_tracker[tailor_step] += (tailor_outer_pde_loss).detach().cpu().item()
                        if opt.operator_loss:
                            tailor_loss = tailor_outer_pde_loss + (tailor_opt_outer_loss)
                        else: 
                            tailor_loss = tailor_outer_pde_loss
                        tailor_loss.backward()
                        # take max norm
                        tailor_param_grads[tailor_step] += max([-1. if p.grad is None else torch.max(torch.abs(p.grad)).item() for p in svg_model.parameters()])
                        if (tailor_step == opt.num_tuning_steps - 1) and (epoch % 10 == 0) and (opt.num_val_batch - 1 == batch_num):
                            gradients = [torch.abs(p.grad).view(-1).tolist() if p.grad is not None else [-1] for p in svg_model.parameters()]
                            for gradient in gradients:
                                grad_collector_all.extend(gradient)
                        inner_opt.step()
                        svg_model.zero_grad(set_to_none=True)
                    
                    with torch.no_grad():
                        gen_seq, mus, logvars, mu_ps, logvar_ps = dont_tailor_many_steps(
                            # no need for higher grads in val
                            svg_model, batch, true_pde_embedding, params, opt=opt, track_higher_grads=False,
                            mode='eval',learnable_model = learnable_model,
                        )
                        val_prediction_collector.append([gen_seq, batch,params[0]])
                        post_opt_outer_loss, post_outer_pde_loss, post_outer_data_loss,post_pde_residual_avg, post_relative_loss, post_log_inner_loss = compute_losses(gen_seq, batch, mus, logvars, mu_ps, logvar_ps, true_pde_embedding, true_pde_embedding, params, opt)
                        val_inner_loss += post_outer_pde_loss.detach().cpu().item() #only log the data loss
                        val_outer_loss += post_outer_data_loss.detach().cpu().item() #only log the data loss
                        val_relative_loss += post_relative_loss.detach().cpu().item()
                        val_post_residual += post_pde_residual_avg.detach().cpu().item()
                        val_log_inner_loss += post_log_inner_loss.detach().cpu().item()
                        for tracker_step in loss_per_step_tracker:
                            if tracker_step not in rel_inner_loss_per_step_tracker:
                                rel_inner_loss_per_step_tracker[tracker_step] = 0
                                rel_loss_per_step_tracker[tracker_step] = 0
                            rel_loss_per_step_tracker[tracker_step] += ((loss_per_step_tracker[tracker_step] - pre_outer_data_loss) / pre_outer_data_loss)#.mean().detach().cpu().item()
                            rel_inner_loss_per_step_tracker[tracker_step]+= ((inner_loss_per_step_tracker[tracker_step] - val_pre_inner_loss) / val_pre_inner_loss)#.mean().detach().cpu().item()
                        val_loss_diff += ((post_outer_data_loss - pre_outer_data_loss) / pre_outer_data_loss).mean().detach().cpu().item()
                        val_inner_loss_diff += ((post_outer_pde_loss - pre_outer_pde_loss) / pre_outer_pde_loss).mean().detach().cpu().item()
                        val_residual_diff += ((post_pde_residual_avg - pre_pde_residual_avg) / pre_pde_residual_avg).mean().detach().cpu().item()
                        # val_prediction_collector.append([gen_seq, batch,params[0]])

                        val_mse_losses.append(post_outer_data_loss.detach().cpu().item())
                        val_rel_losses.append(post_relative_loss.detach().cpu().item())
                        val_pde_losses.append(post_log_inner_loss.detach().cpu().item())


                    if opt.baseline:
                        baseline_outer_loss += base_outer_mse_loss.detach().cpu().item()

                    # writer.add_scalar('val param grads post tailor', np.mean(val_param_grads[-1]),
                        #   (len(val_param_grads)))

            os.makedirs(save_dir, exist_ok = True)
            f = open(f"{save_dir}/losses.txt", "w")
            f.write(f'mse mean:{np.mean([rel_loss for rel_loss in val_mse_losses if math.isfinite(rel_loss)]):.3}+-{np.std([rel_loss for rel_loss in val_mse_losses if math.isfinite(rel_loss)]):.3} \n')
            f.write(f'rel mean:{np.mean([rel_loss for rel_loss in val_rel_losses if math.isfinite(rel_loss)]):.3}+-{np.std([rel_loss for rel_loss in val_rel_losses if math.isfinite(rel_loss)]):.3} \n')
            f.write(f'pde res mean:{np.mean([rel_loss for rel_loss in val_pde_losses if math.isfinite(rel_loss)]):.3}+-{np.std([rel_loss for rel_loss in val_pde_losses if math.isfinite(rel_loss)]):.3} \n')
            f.close()
            save_dict = {'mse_mean':np.mean([rel_loss for rel_loss in val_mse_losses if math.isfinite(rel_loss)]),
                        'rel_mean':np.mean([rel_loss for rel_loss in val_rel_losses if math.isfinite(rel_loss)]),
                        'pde_res_mean':np.mean([rel_loss for rel_loss in val_pde_losses if math.isfinite(rel_loss)]),
                        'param_init_mean':np.mean([rel_loss[-1] for rel_loss in val_param_losses])}
            with open(f'{save_dir}/saved_dictionary.pkl', 'wb') as f:
                pickle.dump(save_dict, f)
            if '1d' in opt.dataset:
            # val_rel_losses = [rel_loss for rel_loss in val_rel_losses if math.isfinite(rel_loss)]
                if 'burgers' not in opt.dataset: 
                    sorted_indices = np.argsort(val_rel_losses)
                    print(sorted_indices)
                    for img_idx, sorted_idx in enumerate(sorted_indices):
                        plt.figure()
                        plt.plot(val_prediction_collector[sorted_idx][0][-1].reshape(-1).detach().cpu().numpy(), label ='ground_truth')
                        plt.plot(val_prediction_collector[sorted_idx][1][-1].reshape(-1).detach().cpu().numpy(), label ='predicition')
                        plt.legend()
                        plt.title(f'relative error: {val_rel_losses[sorted_idx]:0.3}')
                        plt.xlabel('x-spatial coordinates')
                        plt.show()
                        plt.savefig(f"{save_dir}/pred_{img_idx}.jpg")
            else:
                var_1_gt = batch[-1][0,0].detach().cpu().numpy()
                var_1_pred = gen_seq[-1][0,0].detach().cpu().numpy()

                fig, axes = plt.subplots(nrows=3)
                im = axes[0].imshow(var_1_gt)
                axes[0].title.set_text('var 1 gt')
                axes[0].set_xlabel('x_coordinate')
                axes[0].set_ylabel('y_coordinate')
                plt.colorbar(im,ax=axes[0])

                axes[1].imshow(var_1_pred)
                axes[1].title.set_text('var 1 pred')
                axes[1].set_xlabel('x_coordinate')
                axes[1].set_ylabel('y_coordinate')
                plt.colorbar(im,ax=axes[1])
                
                dif = axes[2].imshow(np.abs(var_1_pred - var_1_gt))
                axes[2].title.set_text('absolute difference')
                axes[2].set_xlabel('x_coordinate')
                axes[2].set_ylabel('y_coordinate')
                plt.colorbar(dif,ax=axes[2])

                fig.tight_layout()
                plt.show()
                plt.savefig(f"{save_dir}/var_1")


                var_2_gt = batch[-1][0,1].detach().cpu().numpy()
                var_2_pred = gen_seq[-1][0,1].detach().cpu().numpy()

                fig, axes = plt.subplots(nrows=3)
                im = axes[0].imshow(var_2_gt)
                axes[0].title.set_text('var 2 gt')
                axes[0].set_xlabel('x_coordinate')
                axes[0].set_ylabel('y_coordinate')
                plt.colorbar(im,ax=axes[0])
                    
                axes[1].imshow(var_2_pred)
                axes[1].title.set_text('var 2 pred')
                axes[1].set_xlabel('x_coordinate')
                axes[1].set_ylabel('y_coordinate')
                plt.colorbar(im,ax=axes[1])

                dif = axes[2].imshow(np.abs(var_2_pred - var_2_gt))
                axes[2].title.set_text('absolute difference')
                axes[2].set_xlabel('x_coordinate')
                axes[2].set_ylabel('y_coordinate')
                plt.colorbar(dif,ax=axes[2])

                fig.tight_layout()
                plt.show()
                plt.savefig(f"{save_dir}/var_2")
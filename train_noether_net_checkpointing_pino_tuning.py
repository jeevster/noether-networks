import torch
# This sets the default model weights to float64
# torch.set_default_dtype(torch.float64)  # nopep8
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
import pdb
from models.forward_pino_manual import dont_tailor_many_steps, tailor_many_steps
from models.cn import replace_cn_layers
from models.svg import SVGModel
from models.fno_models import FNOEncoder, FNODecoder
from torch.nn import Linear, Conv2d, ReLU
from models.embedding import ConservedEmbedding, ConvConservedEmbedding, TwoDDiffusionReactionEmbedding
from models.OneD_embeddings import OneDEmbedding
import models.lstm as lstm_models
import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
import ray.util
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch
from neuraloperator.neuralop.models import FNO, FNO1d
from ray.train import RunConfig

from torchinfo import summary
# import hypo
np.float = float
def desired_func(config):
        import torch
        os.chdir('/home/divyam123/noether_work/noether-networks')
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_noether', default=True, type=bool, help='dummy flag indicating we are training the joint noether model. DO NOT CHANGE')
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
        parser.add_argument('--pinn_outer_loss', action='store_true',
                            help='if true, include the (true) PDE residual in outer loss')
        parser.add_argument('--pinn_outer_loss_weight', type=float,
                            default=1.0, help='weight for PDE residual in outer loss')
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
        parser.add_argument('--add_general_learnable', action='store_true')

        # ray.util.pdb.set_trace()
        # NOTE: deterministic for debugging
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        input_params = ['--image_width','128',
                '--g_dim', '128',
                '--z_dim','64',
                '--dataset', '1d_advection_multiparam',
                '--data_root','/data/divyam123/advection_log_space_res_1024',
                '--num_trials','1',
                '--num_threads', '0',
                '--inner_crit_mode', 'mse',
                '--inner_crit_compare_to', 'pde_log',
                '--enc_dec_type','vgg',
                '--num_epochs_per_val','1',
                '--fno_modes', '16',
                '--fno_width', '64',
                '--fno_layers','3',
                '--emb_dim', '64',
                '--num_jump_steps', '0',
                '--n_epochs','200',
                '--outer_opt_model_weights',
                '--random_weights',
                '--only_twenty_degree',
                '--frame_step', '1',
                '--center_crop','1080',
                '--num_emb_frames', '2',
                '--horiz_flip',
                '--reuse_lstm_eps',
                '--num_learned_parameters', '1',
                '--use_partials',
                '--ckpt_outer_loss',
                '--ckpt_inner_loss',
                '--warmstart_emb_path','/home/divyam123/noether_work/noether-networks/best_ckpt_model_advection.pt',
                '--log_dir','/data/divyam123/results_noether_summer/advection/pretrained_embedding_tune_PINO',
                '--channels','1',
                '--random_weights',
                '--inner_opt_all_model_weights',
                '--batch_norm_to_group_norm',
                '--model_path','./checkpoints/pdes/t_past2/batch_5d']
        
        device = torch.device('cuda')
        # --n_past, 2,
        # --n_future, 2,
        # --tailor \
        # --pinn_outer_loss \
        # --emb_type pde_emb \
        # --inner_lr .0001 \
        # --val_inner_lr .0001 \
        # --outer_lr .0001 \
        # --batch_size 8 \ 
        # --num_inner_steps 1 \,learned_pinn_loss,pinn_outer_loss,no_data_loss
        # str(config['emb_type']),
        input_params = input_params + ['--emb_type','pde_const_emb',
                                       '--inner_lr',str(config['inner_lr']),
                                       '--val_inner_lr',str(config['val_inner_lr']),
                                       '--outer_lr',str(config['outer_lr']),
                                       '--batch_size',str(int(config['batch_size'])),
                                       '--num_inner_steps',str(1),
                                       '--n_past',str(2),
                                       '--n_future',str(2)]
        # if config['pinn_outer_loss']:
        # input_params = input_params + ['--pinn_outer_loss']
        # if config['learned_pinn_loss']:
        # input_params = input_params + ['--learned_pinn_loss']
            # else:
            #     if config['no_data_loss']:
            #         input_params = input_params + ['--no_data_loss']
        # pdb.set_trace()
        opt = parser.parse_args(input_params)
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
            tailor_str += f'_{opt.num_inner_steps}'
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
            opt.num_val_batch = len(test_data) #// opt.batch_size

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
        # pdb.set_trace() 

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
            # pdb.set_trace()

            print(f'TRIAL {trial_num}')
            if opt.random_weights:
                print('initializing model with random weights')
                opt.a_dim = 0 if not opt.use_action else opt.a_dim
                # dynamics model
                # FNO1d(n_modes_height = 16, hidden_channels = hidden_channels,
                #     in_channels = in_channels, out_channels = hidden_channels, n_layers = n_layers).to(torch.complex128)
                if opt.dataset in  set(['1d_burgers_multiparam','1d_advection_multiparam','1d_diffusion_reaction_multiparam']):
                    frame_predictor = FNO1d(n_modes_height=opt.fno_modes, hidden_channels=opt.fno_width,
                                        in_channels=opt.channels*opt.n_past, out_channels=opt.channels, n_layers=opt.fno_layers)

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

                # frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim+opt.a_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
                else:
                    print("2d fno")
                    frame_predictor = FNO(n_modes=(opt.fno_modes, opt.fno_modes), hidden_channels=opt.fno_width,
                                        in_channels=opt.channels*opt.n_past, out_channels=opt.channels, n_layers=opt.fno_layers)

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
                # posterior = lstm_models.gaussian_FNO(opt.g_dim+opt.a_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
                # prior = lstm_models.gaussian_FNO(opt.g_dim+opt.a_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size)
                # frame_predictor.apply(utils.init_weights)
                # frame_predictor.apply(utils.init_forget_bias_to_one)
                # posterior.apply(utils.init_weights)
                # prior.apply(utils.init_weights)

                encoder = nn.Identity()  # FNOEncoder()
                decoder = nn.Identity()  # FNOEncoder()


                # encoder = model.encoder(opt.g_dim, opt.channels, use_cn_layers=True, batch_size=opt.batch_size)
                # decoder = model.decoder(opt.g_dim, opt.channels, use_cn_layers=True, batch_size=opt.batch_size)
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
            # load the model from ckpt
            else:
                # loading the last checkpoint
                if opt.reload_latest:
                    restore_checkpoint(svg_model, save_dir, device = device)#torch.device("cuda"))

                # loading the best inner validation model
                elif opt.reload_best_outer:

                    restore_checkpoint(svg_model, save_dir, device = device, best_outer = True, best_inner = False, best_svg = False)
                
                # loading the best outer validation model
                elif opt.reload_best_inner:

                    restore_checkpoint(svg_model, save_dir, device = device, best_outer = False, best_inner = True, best_svg = False)
                
                # loading the best svg validation model
                elif opt.reload_best_svg:

                    restore_checkpoint(svg_model, save_dir, device = device, best_outer = False, best_inner = False, best_svg = True)
                else:
                    svg_model = utils.modernize_model(opt.model_path, opt)
                    print('\nOld SVG model with pre-trained weights loaded and modernized!')

            replace_cn_layers(svg_model.encoder, batch_size=opt.batch_size)
            replace_cn_layers(svg_model.decoder, batch_size=opt.batch_size)
            svg_model.frame_predictor.batch_size = opt.batch_size
            svg_model.posterior.batch_size = opt.batch_size
            svg_model.prior.batch_size = opt.batch_size

            if opt.batch_norm_to_group_norm:
                print('replacing batch norm layers with group norm')
                svg_model = utils.batch_norm_to_group_norm(svg_model)

            #load pretrained embedding model
            if opt.warmstart_emb_path != '' and opt.tailor and opt.emb_type != "pde_const_emb":
                emb_ckpt = torch.load(opt.warmstart_emb_path)
                svg_model.emb.load_state_dict(emb_ckpt['model_state'])

            svg_model.apply(lambda t: t.cuda())
            # print('Eval summary')
            # summary(svg_model.frame_predictor, input_size=(1, opt.n_past*opt.channels, opt.image_width,
            #         opt.image_width), dtypes=[torch.float64], device=torch.device("cuda"),  mode='eval')
            # print('Train summary')
            # summary(svg_model, input_size=(1, opt.n_past*opt.channels, opt.image_width,
            #         opt.image_width), dtypes=[torch.float64], device=torch.device("cuda"), opt=opt, mode='train', i=opt.n_past+2)
            # print('Emb summary')
            # #summary(svg_model.emb, input_size=(opt.n_past, opt.channels, opt.image_width, opt.image_width), device=torch.device("cuda"))
            # summary(svg_model.emb, input_size=(1, opt.num_emb_frames * opt.channels, opt.image_width, opt.image_width), dtypes=[torch.float64], device=torch.device("cuda"))
            # For comparing later
            old_state_dict = copy.deepcopy(svg_model.state_dict())
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

            baseline_outer_losses = []
            outer_losses = []
            svg_losses = []
            val_svg_losses = []
            inner_losses = []
            true_inner_losses = []
            val_outer_losses = []
            val_inner_losses = []
            val_true_inner_losses = []
            emb_weights = []
            emb_biases = []
            all_gen = None
            param_grads = []
            grad_norms = []
            emb_norms = []

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
            for epoch in range(start_epoch, opt.n_epochs):

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
                if epoch % opt.num_epochs_per_val == 0:
                    print(f'Validation {epoch} Epoch')
                    val_outer_loss = 0.
                    baseline_outer_loss = 0.

                    svg_model.eval()

                    for batch_num in tqdm(range(opt.num_val_batch)):
                        # pdb.set_trace()
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
                                base_outer_mse_loss, base_outer_pde_loss = svg_crit(base_gen_seq, batch, base_mus, base_logvars,
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
                        for batch_step in range(opt.num_jump_steps + 1):
                            # jump steps are effectively inner steps that have a single higher innerloop_ctx per
                            # iteration, which allows for many inner steps during training without running
                            # into memory issues due to storing the whole dynamic computational graph
                            # associated with unrolling the sequence in the inner loop for many steps

                            # perform tailoring (autoregressive prediction, tailoring, predict again)
                            # pdb.set_trace()
                            # Tailor many steps uses opt.tailor to decide whether to tailor or not
                            # val_inner_lr = 0.0001
                            gen_seq, mus, logvars, mu_ps, logvar_ps = tailor_many_steps(
                                # no need for higher grads in val
                                svg_model, batch, true_pde_embedding, params, opt=opt, track_higher_grads=False,
                                mode='eval',learnable_model = learnable_model,
                                # extra kwargs
                                tailor_losses=val_batch_inner_losses,
                                true_tailor_losses = val_batch_true_inner_losses,
                                param_losses = val_batch_inner_param_losses,
                                true_param_losses = val_batch_true_inner_param_losses,
                                inner_gain = val_batch_inner_gain,
                                true_inner_gain = val_batch_true_inner_gain, 
                                inner_crit_mode=opt.inner_crit_mode,
                                reuse_lstm_eps=opt.reuse_lstm_eps,
                                val_inner_lr=val_inner_lr,
                                svg_losses=val_batch_svg_losses,
                                only_cn_decoder=opt.only_cn_decoder,
                                # fstate_dict=val_fstate_dict,
                                cached_cn=val_cached_cn,
                                load_cached_cn=(batch_step != 0),
                            )

                            with torch.no_grad():
                                # compute outer (task) loss
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


                            val_outer_loss += outer_mse_loss.detach().cpu().item() #only log the data loss
                            if opt.baseline:
                                baseline_outer_loss += base_outer_mse_loss.detach().cpu().item()

                        #SR: want to log inner losses for all tailoring steps, not just the first step
                        val_batch_inner_losses = val_batch_inner_losses[0]
                        val_batch_inner_param_losses = val_batch_inner_param_losses[0]
                        val_batch_svg_losses = val_batch_svg_losses[0]
                        val_batch_inner_gain = val_batch_inner_gain[0] #if opt.tailor else val_batch_true_inner_gain
                        if len(val_batch_true_inner_losses) !=0:
                            val_batch_true_inner_losses = val_batch_true_inner_losses[0]
                            val_batch_true_inner_param_losses = val_batch_true_inner_param_losses[0]
                            val_batch_true_inner_gain = val_batch_true_inner_gain[0] #if opt.tailor else val_batch_true_inner_gain
                        # if (opt.num_inner_steps > 0 or opt.num_jump_steps > 0) and opt.tailor:
                        #     # fix the inner losses to account for jump step
                        #     # after the zeroth, take the tailored inner loss (index 1)
                        #     val_batch_svg_losses = [
                        #         val_batch_svg_losses[0][0]] + [l[1] for l in val_batch_svg_losses]
                        #     val_batch_inner_losses = [val_batch_inner_losses[0][0]] + [l[1] for l in val_batch_inner_losses]
                        # else:
                        #     val_batch_svg_losses = [val_batch_svg_losses[0][0]]
                        #     #if opt.tailor:
                        #     val_batch_inner_losses = [val_batch_inner_losses[0][0]]
                        #     # else:
                        #     #     val_batch_inner_losses = [
                        #     #         0 for _ in range(len(val_batch_svg_losses))]

                        # Should be zero when opt.tailor is False
                        # pdb.set_trace()
                        if len(val_batch_true_inner_losses) !=0:
                            epoch_val_true_inner_losses.append(val_batch_true_inner_losses)
                            epoch_val_true_param_losses.append(val_batch_true_inner_param_losses)
                            epoch_val_true_inner_gain.append(val_batch_true_inner_gain)
                        epoch_val_inner_losses.append(val_batch_inner_losses)
                        epoch_val_param_losses.append(val_batch_inner_param_losses)
                        epoch_val_inner_gain.append(val_batch_inner_gain)
                        epoch_val_svg_losses.append(val_batch_svg_losses)
                    # pdb.set_trace()
                    if len(val_batch_true_inner_losses) !=0:
                        val_true_inner_losses.append([sum(x) / (opt.num_val_batch)
                                            for x in zip(*epoch_val_true_inner_losses)])
                        val_true_param_losses.append([sum(x) / (opt.num_val_batch)
                                            for x in zip(*epoch_val_true_param_losses)])
                        # val_true_inner_gain.append([sum(x) / (opt.num_val_batch)
                                            # for x in  zip(*epoch_val_true_inner_gain)])
                        val_true_inner_gain.append(sum(epoch_val_true_inner_gain) / (opt.num_val_batch))

                    # val_inner_gain.append([sum(x) / (opt.num_val_batch)
                    #                         for x in zip(*epoch_val_inner_gain)])
                    val_inner_gain.append(sum(epoch_val_inner_gain) / (opt.num_val_batch))

                    val_inner_losses.append([sum(x) / (opt.num_val_batch)
                                            for x in zip(*epoch_val_inner_losses)])
                    val_param_losses.append([sum(x) / (opt.num_val_batch)
                                            for x in zip(*epoch_val_param_losses)])

                    val_svg_losses.append([sum(x) / (opt.num_val_batch)
                                        for x in zip(*epoch_val_svg_losses)])
                    val_outer_losses.append(val_outer_loss / (opt.num_val_batch))
                    train.report({'loss':val_outer_losses[-1]})
                    if opt.baseline:
                        baseline_outer_losses.append(
                            baseline_outer_loss / (opt.num_val_batch))
                    # pdb.set_trace()
                    writer.add_scalar('Outer Loss/val', val_outer_losses[-1],
                                    (epoch + 1))
                    if opt.baseline:
                        writer.add_scalar('Outer Loss/baseline', baseline_outer_losses[-1],
                                        (epoch + 1))
                        if opt.verbose:
                            print(f'\tOuter BASE loss:  {baseline_outer_losses[-1]}')
                    if opt.tailor:
                        for step, value in enumerate(val_inner_losses[-1]):
                            print("LOGGING INNER PDE LOSS", len(val_inner_losses[-1]))
                            writer.add_scalar(
                                f'Inner Loss/val/{step} Step', value, (epoch + 1))
                            writer.add_scalar(
                                f'Inner Param Loss/val/{step} Step', val_param_losses[-1][step], (epoch + 1))

                    # if opt.tailor:
                        writer.add_scalar(
                            f'Inner Loss Gain/val', val_inner_gain[-1], (epoch + 1))
                    #Log true PDE loss
                    try:
                        for step, value in enumerate(val_true_inner_losses[-1]):
                            writer.add_scalar(
                                f'True Inner Loss/val/{step} Step', value, (epoch + 1))
                            writer.add_scalar(
                                f'True Inner Param Loss/val/{step} Step', val_true_param_losses[-1][step], (epoch + 1))
                        # if opt.tailor:
                            writer.add_scalar(
                                f'True Inner Loss Gain/val', val_true_inner_gain[-1], (epoch + 1))
                    except IndexError:
                        pass
                    if opt.verbose:
                        print(f'\tInner VAL loss:   {val_inner_losses[-1]}')
                    for step, value in enumerate(val_svg_losses[-1]):
                        writer.add_scalar(
                            f'SVG Loss/val/{step} Step', value, (epoch + 1))
                    if opt.verbose:
                        print(f'\tSVG VAL loss:     {val_svg_losses[-1]}')
                    writer.flush()
                #checkpointing best val
                if opt.save_checkpoint:
                    if opt.ckpt_outer_loss:
                        if  min_val_outer_loss > val_outer_losses[-1]:
                            min_val_outer_loss = val_outer_losses[-1]
                            save_checkpoint(embedding, save_dir, True, False, False)
                    if opt.ckpt_inner_loss:
                        if  min_val_inner_loss > val_inner_losses[-1][-1]:
                            min_val_inner_loss = val_inner_losses[-1][-1]
                            save_checkpoint(embedding, save_dir, False, True, False)        
                    if opt.ckpt_svg_loss:
                        if  min_val_svg_loss > val_svg_losses[-1][-1]:
                            min_val_svg_loss = val_svg_losses[-1][-1]
                            save_checkpoint(embedding, save_dir, False, False, True)    
                # Training
                print(f'Train {epoch} Epoch')

                for batch_num in tqdm(range(opt.num_train_batch)):
                    batch, params = next(training_batch_generator)
                    params = tuple([param.to(torch.cuda.current_device()) for param in params])
                    train_mode = 'eval' if opt.no_teacher_force else 'train'
                    # tailoring pass
                    cached_cn = [None]  # cached cn params
                    batch_inner_losses = []
                    batch_true_inner_losses = []
                    batch_svg_losses = []
                    batch_inner_param_losses = []
                    batch_true_inner_param_losses = []
                    batch_inner_gain = []
                    batch_true_inner_gain = []
                    # pdb.set_trace()
                    # val_inner_lr = 0
                    for batch_step in range(opt.num_jump_steps + 1):

                        # rollout, tailoring, rollout
                        # Again, tailor_many_steps uses opt.tailor to decide whether to tailor or not
                        gen_seq, mus, logvars, mu_ps, logvar_ps = dont_tailor_many_steps(
                            svg_model, batch, true_pde_embedding, params, opt=opt, track_higher_grads=True,
                            mode=train_mode, learnable_model = learnable_model,
                            # extra kwargs
                            tailor_losses=batch_inner_losses,
                            true_tailor_losses = batch_true_inner_losses,
                            param_losses=batch_inner_param_losses,
                            true_param_losses = batch_true_inner_param_losses,
                            inner_gain = batch_inner_gain,
                            true_inner_gain = batch_true_inner_gain, 
                            inner_crit_mode=opt.inner_crit_mode,
                            reuse_lstm_eps=opt.reuse_lstm_eps,
                            svg_losses=batch_svg_losses,
                            only_cn_decoder=opt.only_cn_decoder,
                            # fstate_dict=train_fstate_dict,
                            cached_cn=cached_cn,
                            load_cached_cn=(batch_step != 0),
                        )

                        # compute task loss
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

                        # don't do this
                        if opt.add_inner_to_outer_loss:
                            inner_loss_component = inner_crit(svg_model, gen_seq, mode='mse',
                                                            num_emb_frames=opt.num_emb_frames,
                                                            compare_to=opt.inner_crit_compare_to).mean()
                            print(
                                f'outer_loss = {outer_mse_loss.detach().cpu().numpy().item()}')
                            print(
                                f'inner_loss = {inner_loss_component.detach().cpu().numpy().item()}')
                            outer_loss += inner_loss_component

                        train_outer_loss += outer_mse_loss.detach().cpu().item() #only keep data loss for logging

                        # Compute gradients of task loss (including both PDE and MSE loss)
                        outer_loss.backward()
                        if opt.num_inner_steps > 0:
                            # gradient clipping, and tracking the grad norms
                            param_grads.append(
                                [-1. if p.grad is None else torch.norm(p.grad).item() for p in svg_model.parameters()])
                            grad_norm = nn.utils.clip_grad_norm_(svg_model.emb.parameters(), 1000)
                            grad_norm_sum += grad_norm.item()
                            emb_p = [torch.norm(p.detach())
                                    for p in svg_model.emb.parameters()]
                            if len(emb_p) > 0:
                                emb_norm_sum += torch.norm(torch.stack(emb_p)).item()
                    #SR: want to log inner losses for all tailoring steps, not just the first step
                    if len(batch_true_inner_losses) != 0:
                        batch_true_inner_losses = batch_true_inner_losses[0]
                        batch_true_inner_param_losses = batch_true_inner_param_losses[0]
                        batch_true_inner_gain = batch_true_inner_gain[0] #if opt.tailor else batch_true_inner_gain
                    batch_inner_param_losses = batch_inner_param_losses[0]
                    batch_inner_losses = batch_inner_losses[0]
                    batch_inner_gain = batch_inner_gain[0] #if opt.tailor else batch_inner_gain
                    batch_svg_losses = batch_svg_losses[0]
                    # if (opt.num_inner_steps > 0 or opt.num_jump_steps > 0) and opt.tailor:
                    #     # fix the inner losses to account for jump step
                    #     batch_inner_losses = [batch_inner_losses[0]
                    #                           [0]] + [l[1] for l in batch_inner_losses]
                    #     batch_svg_losses = [batch_svg_losses[0][0]
                    #                         ] + [l[1] for l in batch_svg_losses]
                    # else:
                    #     batch_svg_losses = [batch_svg_losses[0][0]]
                    #     #if opt.tailor:
                    #     batch_inner_losses = [batch_inner_losses[0][0]]
                    #     # else:
                    #     #     batch_inner_losses = [
                    #     #         0 for _ in range(len(batch_svg_losses))]
                    if len(batch_true_inner_losses) != 0:
                        epoch_true_inner_losses.append(batch_true_inner_losses)
                        epoch_true_param_losses.append(batch_true_inner_param_losses)
                        epoch_true_inner_gain.append(batch_true_inner_gain)
                    epoch_param_losses.append(batch_inner_param_losses)
                    epoch_inner_losses.append(batch_inner_losses)
                    epoch_inner_gain.append(batch_inner_gain)
                    epoch_svg_losses.append(batch_svg_losses)

                    # Update conservation model and task model
                    outer_opt.step()
                    svg_model.zero_grad(set_to_none=True)

                # pdb.set_trace()
                svg_losses.append([sum(x) / (opt.num_train_batch)
                                for x in zip(*epoch_svg_losses)])
                inner_losses.append([sum(x) / (opt.num_train_batch)
                                    for x in zip(*epoch_inner_losses)])
                param_losses.append([sum(x) / (opt.num_train_batch)
                                    for x in zip(*epoch_param_losses)])
                # inner_gain.append([sum(x) / (opt.num_train_batch)
                #                     for x in zip(*epoch_inner_gain)])
                inner_gain.append(sum(epoch_inner_gain) / (opt.num_train_batch))
                if len(batch_true_inner_losses) != 0:
                    true_inner_losses.append([sum(x) / (opt.num_train_batch)
                                    for x in zip(*epoch_true_inner_losses)])
                    true_param_losses.append([sum(x) / (opt.num_train_batch)
                                    for x in zip(*epoch_true_param_losses)])
                    
                    true_inner_gain.append(sum(epoch_true_inner_gain) / (opt.num_train_batch))
                    # true_inner_gain.append([sum(x) / (opt.num_train_batch)
                    #                 for x in zip(*epoch_true_inner_gain)])

                grad_norms.append(grad_norm_sum / opt.num_train_batch)
                emb_norms.append(emb_norm_sum / opt.num_train_batch)

                outer_losses.append(train_outer_loss / (opt.num_train_batch))
                writer.add_scalar('Outer Loss/train', outer_losses[-1],
                                (epoch + 1))
                if opt.tailor:
                    for step, value in enumerate(inner_losses[-1]):
                        writer.add_scalar(
                            f'Inner Loss/train/{step} Step', value, (epoch + 1))
                        writer.add_scalar(
                            f'Inner Param Loss/train/{step} Step', param_losses[-1][step], (epoch + 1))

                # if opt.tailor:
                    writer.add_scalar(
                        f'Inner Loss Gain/train', inner_gain[-1], (epoch + 1))
                try:
                    for step, value in enumerate(true_inner_losses[-1]):
                        writer.add_scalar(
                            f'True Inner Loss/train/{step} Step', value, (epoch + 1))
                        writer.add_scalar(
                            f'True Inner Param Loss/train/{step} Step', true_param_losses[-1][step], (epoch + 1))
                        
                    # if opt.tailor:
                    writer.add_scalar(
                        f'True Inner Loss Gain/train', true_inner_gain[-1], (epoch + 1))
                except IndexError:
                    pass

                if opt.verbose:
                    print(f'\tInner TRAIN loss: {inner_losses[-1]}')
                for step, value in enumerate(svg_losses[-1]):
                    writer.add_scalar(
                        f'SVG Loss/train/{step} Step', value, (epoch + 1))

                if opt.verbose:
                    print(f'\tSVG TRAIN loss: {svg_losses[-1]}')
                if opt.tailor:
                    writer.add_scalar('Embedding/grad norm', grad_norms[-1],
                                    (epoch + 1))
                    writer.add_scalar('Embedding/param norm', emb_norms[-1],
                                    (epoch + 1))
                if opt.learn_inner_lr:
                    writer.add_scalar('Embedding/inner LR scale factor', svg_model.inner_lr_scale,
                                    (epoch + 1))
                    all_inner_lr_scales.append(
                        svg_model.inner_lr_scale.detach().cpu().item())

                # checkpointing
                if opt.save_checkpoint:
                    save_checkpoint(svg_model,save_dir)

            all_outer_losses.append(copy.deepcopy(svg_losses))
            all_inner_losses.append(copy.deepcopy(inner_losses))
            all_val_outer_losses.append(copy.deepcopy(val_svg_losses))
            all_baseline_outer_losses.append(copy.deepcopy(baseline_outer_losses))
            all_val_inner_losses.append(copy.deepcopy(val_inner_losses))
            all_emb_weights.append(copy.deepcopy(emb_weights))
            all_emb_biases.append(copy.deepcopy(emb_biases))
            all_param_grads.append(copy.deepcopy(param_grads))
            all_grad_norms.append(copy.deepcopy(grad_norms))
            all_emb_norms.append(copy.deepcopy(emb_norms))

        # Log hyper parameters and final losses
        final_metrics = {}
        metrics = [
            'Outer Loss/train',
            'Outer Loss/val',
            'Inner Loss/train',
            'Inner Loss/val',
        ]
        metric_lists = [
            all_outer_losses,
            all_val_outer_losses,
            all_inner_losses,
            all_val_inner_losses,
        ]

        #Note: still want to keep track of inner loss even when we're not tailoring
        # if not opt.tailor:
        #     metrics.pop()
        #     metrics.pop()
        #     metric_lists.pop()
        #     metric_lists.pop()

        for metric, metric_list in zip(metrics, metric_lists):
            # grab the last loss
            final_metrics['Best ' + metric] = min(*[i[-1]
                                                for j in metric_list for i in j])

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
            'batch_norm_to_group_norm': opt.batch_norm_to_group_norm
        }
        hparams_logging = hparams(hyperparameters, final_metrics)
        for i in hparams_logging:
            writer.file_writer.add_summary(i)
        for k, v in final_metrics.items():
            writer.add_scalar(f'{k}', v)
        writer.flush()
        writer.close()

        if opt.verbose:
            print(5*'\n')
            print('all_inner_losses')
            print(all_inner_losses)
            print('all_val_inner_losses')
            print(all_val_inner_losses)
            print('all_outer_losses')
            print(all_outer_losses)
            print('all_val_outer_losses')
            print(all_val_outer_losses)
            print('all_baseline_outer_losses')
            print(all_baseline_outer_losses)
            print('all_grad_norms')
            print(all_grad_norms)




def main():

    config = {
        "inner_lr": tune.loguniform(1e-5,1e-1),
        "val_inner_lr": tune.loguniform(1e-5,1e-1),
        "outer_lr": tune.loguniform(1e-5,1e-1),
        # "inner_lr": tune.choice([1e-5,1e-4,1e-3]),
        # "val_inner_lr": tune.choice([1e-5,1e-4,1e-3]),
        # "outer_lr": tune.choice([1e-5,1e-4,1e-3]),
        # "num_inner_steps": tune.choice([1,2,5]),
        "batch_size":tune.choice([1,2,4,8,16]),
        # "n_future":tune.choice([i for i in range(2,11)]),
        # "n_past":tune.choice([i for i in range(2,11)]),
        # "pinn_outer_loss":tune.choice([True, False]),
        # "no_data_loss":tune.choice([True, False]),
        # "learned_pinn_loss":tune.choice([True, False]),
        # "emb_type":tune.choice(['pde_emb', 'pde_const_emb']),
    }
    # config = {
    #     "inner_lr": tune.choice([1e-5,1e-4,1e-3,1e-2,1e-1]),
    #     "val_inner_lr": tune.choice([1e-5,1e-4,1e-3,1e-2,1e-1]),
    #     "outer_lr": tune.choice([1e-5,1e-4,1e-3,1e-2,1e-1]),
    #     "num_inner_steps": tune.choice([1,2,5]),
    #     "batch_size":tune.choice([4,8,16]),
    #     "n_future":tune.choice([i for i in range(1,11)]),
    #     "n_past":tune.choice([i for i in range(1,11)]),
    #     "pinn_outer_loss":tune.choice([True, False]),
    #     "learned_pinn_loss":tune.choice([True, False]),
    #     "emb_type":tune.choice(['pde_emb', 'pde_const_emb']),
    # }

    algo = OptunaSearch()#utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
    # algo = ConcurrencyLimiter(algo, max_concurrent=1)

    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=200,
        grace_period=50,
        reduction_factor=4,
        brackets=1,
    )
    # pdb.set_trace()
    tuner = tune.Tuner(
        # desired_func,
        tune.with_resources(desired_func, resources={"gpu": 1}),
        tune_config=tune.TuneConfig(
                    metric = "loss",
                    mode = "min",
                    search_alg = algo,
                    num_samples = 35,
                    scheduler = asha_scheduler
                ),
                param_space = config,
                run_config = RunConfig(
                                    storage_path = "/data/divyam123/ray_tune_experiments",
                                    name = "data_true_(outer)_true(inner)_tuning",
                                )
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    # print("Best trial final validation accuracy: {}".format(
    #     best_result.metrics["accuracy"]))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    # ray.init(_temp_dir='.')
    main()
    # desired_func({'inner_lr':str(1e-4),'outer_lr':str(1e-4),'val_inner_lr':str(1e-4)})

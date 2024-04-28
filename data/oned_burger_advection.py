import os
import torch
import torchvision
import random
import math
import h5py
import numpy as np
import glob
import pdb


class CropUpperRight(torch.nn.Module):
    def __init__(self, w):
        super(CropUpperRight, self).__init__()
        self.w = w

    def forward(self, img):
        return img[:,:,:self.w,-self.w:]
"""new loader"""
class OneD_Advection_Burgers_MultiParam(object):
    """Data Loader that loads multiple parameter version of Advection or Burgers ReacDiff dataset"""

    def __init__(self, pde ,data_root, train=True,
                 seq_len=101, image_size=128, length=-1, percent_train=.8,
                 frame_step=1,
                 shuffle=True, num_param_combinations=-1, fixed_ic = False, fixed_window = False, fno_rnn = False, ood = False):
        '''
        if length == -1, use all sequences available
        '''
        self.pde = pde
        self.data_root = data_root 
        self.seq_len = seq_len
        self.image_size = image_size
        self.frame_step = frame_step
        self.length = length
        self.train = train
        self.h5_paths = glob.glob(f"{self.data_root}/*.hdf5")
        self.h5_files = [h5py.File(file, "r") for file in self.h5_paths]
        self.seqs = [h5py.File(file, "r") for file in self.h5_paths]#self.h5_files
        self.num_param_combinations = num_param_combinations
        self.fixed_ic = fixed_ic
        self.fixed_window = fixed_window
        self.fno_rnn = fno_rnn
        if shuffle:
            print('shuffling dataset')
            random.Random(1612).shuffle(self.seqs)
            random.Random(1612).shuffle(self.h5_paths)
            random.Random(1612).shuffle(self.h5_files)
        if num_param_combinations > 0:
            print(f'trimming dataset from length {len(self.seqs)} to {num_param_combinations}')
            self.seqs = self.seqs[:num_param_combinations]
            self.h5_paths = self.h5_paths[:num_param_combinations]
            self.h5_files = self.h5_files[:num_param_combinations]
        if ood:
            # 0.028849327853371208, 0.6926961865675941
            if train:
                self.seqs_temp = []
                self.h5_paths_temp = []
                self.h5_files_temp = []
                for idx,path in enumerate(self.h5_paths):
                    if self.extract_params_from_path(path)[0] >= 0.028849327853371208 and self.extract_params_from_path(path)[0] <= 0.6926961865675941:
                        self.seqs_temp.append(self.seqs[idx])
                        self.h5_paths_temp.append(self.h5_paths[idx])
                        self.h5_files_temp.append(self.h5_files[idx])
                self.seqs = self.seqs_temp
                self.h5_paths = self.h5_paths_temp
                self.h5_files = self.h5_files_temp
            else:
                self.seqs_temp = []
                self.h5_paths_temp = []
                self.h5_files_temp = []
                for idx,path in enumerate(self.h5_paths):
                    if self.extract_params_from_path(path)[0] < 0.028849327853371208 and self.extract_params_from_path(path)[0] > 0.6926961865675941:
                        self.seqs_temp.append(self.seqs[idx])
                        self.h5_paths_temp.append(self.h5_paths[idx])
                        self.h5_files_temp.append(self.h5_files[idx])
                self.seqs = self.seqs_temp
                self.h5_paths = self.h5_paths_temp
                self.h5_files = self.h5_files_temp
        else:
            if train:
                self.seqs = self.seqs[:int(len(self.seqs)*percent_train)]
                self.h5_paths = self.h5_paths[:int(len(self.h5_paths)*percent_train)]
                self.h5_files = self.h5_files[:int(len(self.h5_files)*percent_train)]
            else:
                self.seqs = self.seqs[int(len(self.seqs)*percent_train):]
                self.h5_paths = self.h5_paths[int(len(self.h5_paths)*percent_train):]
                self.h5_files = self.h5_files[int(len(self.h5_files)*percent_train):]


        print(f"Initialized {'train' if train else 'val'} dataset with {len(self.seqs)} parameter combinations")
    def extract_params_from_path(self, path):
        if self.pde == 'burgers':
            params = [float(path[:-5].split('u')[-1])]
        elif self.pde == 'advection':
            params = [float(path[:-5].split('a')[-1])]
        elif self.pde == 'diffuion_reaction':
            nu = float(path.split('_')[-3])
            rho = float(path.split('_')[-1][:-5])
            params = [rho, nu]
        return params


    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, index):
        param = index
        seqs = self.seqs[param] # choose a file (i.e parameter value) from 1372 possibilies
        random_index = np.random.choice(len(seqs['tensor']))
        seq = seqs['tensor'][0] if self.fixed_ic else seqs['tensor'][random_index]
        params = self.extract_params_from_path(self.h5_paths[param])
        #get data        
        vid = torch.Tensor(seq).to(torch.cuda.current_device())
        #sample a random window from this trajectory starting at 20 to get rid of high residuals (101 - 20 - self.seq_len possibilities)
        
        start = 50 if self.fixed_window else np.random.randint(50, vid.shape[0] - self.seq_len)
        vid = vid[start:start+self.seq_len] if self.seq_len != -1 else vid[start:-1]
        vid = vid.reshape((vid.shape[0], 1,vid.shape[1], 1))
        
        if self.frame_step > 1:
            vid = vid[::self.frame_step]  # only take the video frames
        return vid, tuple(params)


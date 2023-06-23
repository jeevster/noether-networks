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

    def __init__(self, burgers ,data_root, train=True,
                 seq_len=101, image_size=128, length=-1, percent_train=.8,
                 frame_step=1,
                 shuffle=False, num_param_combinations=-1, fixed_ic = False, fixed_window = False):
        '''
        if length == -1, use all sequences available
        '''
        self.pde = burgers
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
        if self.pde == True:
            nu = float(path[:-5].split('u')[-1])
        else:
            nu = float(path[:-5].split('a')[-1])
        return nu


    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, index):
        # pdb.set_trace()        
        param = index
        seqs = self.seqs[param] # choose a file (i.e parameter value) from 1372 possibilies
        random_index = np.random.choice(len(seqs['tensor']))
        seq = seqs['tensor'][0] if self.fixed_ic else seqs['tensor'][random_index]#[0]
        nu = self.extract_params_from_path(self.h5_paths[param])
        #get data        
        vid = torch.Tensor(seq).to(torch.float64).to(torch.cuda.current_device())
        #sample a random window from this trajectory starting at 20 to get rid of high residuals (101 - 20 - self.seq_len possibilities)
        start = 50 if self.fixed_window else np.random.randint(50, vid.shape[0] - self.seq_len)
        vid = vid[start:start+self.seq_len]
        vid = vid.reshape((vid.shape[0], vid.shape[1], 1))
        
        if self.frame_step > 1:
            vid = vid[::self.frame_step]  # only take the video frames
        return vid, tuple([nu])



# """old loader"""
# class OneD_Advection_Burgers_MultiParam(object):
#     """Data Loader that loads multiple parameter version of Advection or Burgers ReacDiff dataset"""

#     def __init__(self, burgers ,data_root, train=True,
#                  seq_len=101, image_size=128, length=-1, percent_train=.8,
#                  frame_step=1,
#                  shuffle=False, num_param_combinations=-1, fixed_ic = False, fixed_window = False):
#         '''
#         if length == -1, use all sequences available
#         '''
#         self.pde = burgers
#         self.data_root = data_root 
#         self.seq_len = seq_len
#         self.image_size = image_size
#         self.frame_step = frame_step
#         self.length = length
#         self.train = train
#         self.h5_paths = glob.glob(f"{self.data_root}/*.hdf5")
#         self.h5_files = [(h5py.File(file, "r"),file) for file in self.h5_paths]
#         self.seqs = [(h5py.File(file, "r"),file) for file in self.h5_paths]
#         self.num_param_combinations = num_param_combinations
#         self.fixed_ic = fixed_ic
#         self.fixed_window = fixed_window
#         if shuffle:
#             print('shuffling dataset')
#             random.Random(1612).shuffle(self.seqs)
#         if length > 0:
#             print(f'trimming dataset from length {len(self.seqs)} to {length}')
#             self.seqs = [(seq['tensor'][:length], file) for seq,file in self.seqs]
#         if train:
#             # pdb.set_trace()
#             self.seqs = [(seq['tensor'][:int((30)*percent_train)], file) for seq,file in self.seqs]
#         else:
#             self.seqs = [(seq['tensor'][int((30)*percent_train):30], file) for seq,file in self.seqs]

#         print(f"Initialized {'train' if train else 'val'} dataset with {len(self.seqs)} parameter combinations")
#     def extract_params_from_path(self, path):
#         if self.pde == True:
#             nu = float(path[:-5].split('u')[-1])
#         else:
#             nu = float(path[:-5].split('a')[-1])
#         return nu


#     def __len__(self):
#         if self.train:
#             return self.num_param_combinations if self.num_param_combinations > 0 else len(self.seqs) #* 4 # only 20 param combinations - test overfitting
#         else:
#             return int(self.num_param_combinations/4) if self.num_param_combinations > 0 else int(len(self.seqs)/4) #* 4 # number of [parameter, trajectory, window] combinations we see per epoch
#     def __getitem__(self, index):
#         # pdb.set_trace()
#         param = index
#         seqs = self.seqs[param][0] # choose a file (i.e parameter value) from 1372 possibilies
#         random_index = np.random.choice(len(seqs))#len(seqs['tensor']))

#         seq = seqs[0] if self.fixed_ic else seqs[random_index]

#         nu = self.extract_params_from_path(self.seqs[param][1])
        

#         vid = torch.Tensor(seq).to(torch.float64).to(torch.cuda.current_device())
#         #sample a random window from this trajectory starting at 20 to get rid of high residuals (101 - 20 - self.seq_len possibilities)
#         start = 20 if self.fixed_window else np.random.randint(20, vid.shape[0] - self.seq_len)
#         vid = vid[start:start+self.seq_len]
#         vid = vid.reshape((vid.shape[0], vid.shape[1], 1))
        
#         if self.frame_step > 1:
#             vid = vid[::self.frame_step]  # only take the video frames
#         return vid, tuple([nu])
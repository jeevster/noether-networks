import os
import torch
import torchvision
import random
import math
import h5py
import numpy as np
import glob


class CropUpperRight(torch.nn.Module):
    def __init__(self, w):
        super(CropUpperRight, self).__init__()
        self.w = w

    def forward(self, img):
        return img[:,:,:self.w,-self.w:]

class TwoDReacDiff(object):
    """Data Loader that loads ReacDiff 101."""

    def __init__(self, data_root, train=True,
                 seq_len=101, image_size=128, length=-1, percent_train=.8,
                 frame_step=1,
                 shuffle=False):
        '''
        if length == -1, use all sequences available
        '''
        self.data_root = data_root 
        self.seq_len = seq_len
        self.image_size = image_size
        self.frame_step = frame_step
        self.idx = 0
        self.h5_file = h5py.File(os.path.join(self.data_root, "2D_diff-react_NA_NA.h5"), "r")
        self.seqs = list(self.h5_file.keys())


        if shuffle:
            print('shuffling dataset')
            random.Random(1612).shuffle(self.seqs)
        if length > 0:
            print(f'trimming dataset from length {len(self.seqs)} to {length}')
            self.seqs = self.seqs[:length]
        if train:
            self.seqs = self.seqs[:int(len(self.seqs)*percent_train)]
        else:
            self.seqs = self.seqs[int((len(self.seqs)+1)*percent_train):]
        
        print(f"Initialized {'train' if train else 'test'} dataset with {len(self.seqs)} examples")

    def __len__(self):
        return len(self.seqs)
              
    def __getitem__(self, index):
        seq = self.seqs[self.idx]

        if self.idx == len(self.seqs) - 1:
            #loop back to beginning
            self.idx = 0
        else:
            self.idx+=1

        
        #get data
        vid = torch.Tensor(np.array(self.h5_file[f"{seq}/data"], dtype="f")).to(torch.cuda.current_device()) # dim = [101, 128, 128, 2]

        #sample a random window
        start  = np.random.randint(0, vid.shape[0] - self.seq_len)
        vid = vid[start:start+self.seq_len]
        
        if self.frame_step > 1:
            vid = vid[::self.frame_step]  # only take the video frames

        return vid


class TwoDReacDiff_MultiParam(object):
    """Data Loader that loads multiple parameter version of 2D ReacDiff dataset"""

    def __init__(self, data_root, train=True,
                 seq_len=101, image_size=128, length=-1, percent_train=.8,
                 frame_step=1,
                 shuffle=False, num_param_combinations=-1, fixed_ic = False, fixed_window = False):
        '''
        if length == -1, use all sequences available
        '''
        self.data_root = data_root 
        self.seq_len = seq_len
        self.image_size = image_size
        self.frame_step = frame_step
        self.length = length
        self.train = train
        self.h5_paths = glob.glob(f"{self.data_root}/*.h5")
        self.h5_files = [h5py.File(file, "r") for file in self.h5_paths]
        self.seqs = [list(h5_file.keys()) for h5_file in self.h5_files]
        self.num_param_combinations = num_param_combinations
        self.fixed_ic = fixed_ic
        self.fixed_window = fixed_window


        if shuffle:
            print('shuffling dataset')
            random.Random(1612).shuffle(self.seqs)
        if length > 0:
            print(f'trimming dataset from length {len(self.seqs)} to {length}')
            self.seqs = [seq[:length] for seq in self.seqs]
        if train:
            self.seqs = [seq[:int(len(seq)*percent_train)] for seq in self.seqs]
        else:
            self.seqs = [seq[int((len(seq))*percent_train):] for seq in self.seqs]
        
        print(f"Initialized {'train' if train else 'test'} dataset with {len(self.seqs)} examples")

    def extract_params_from_path(self, path):
        name = os.path.basename(path)
        elements = name.split("=")[1:]
        du = float(elements[0].split("_")[0])
        dv = float(elements[1].split("_")[0])
        k = float(elements[2].split(".")[0] + "." + elements[2].split(".")[1])
        return k, du, dv


    def __len__(self):
        if self.train:
            return self.num_param_combinations if self.num_param_combinations > 0 else len(self.seqs)# only 20 param combinations - test overfitting
        else:
            return int(self.num_param_combinations/4) if self.num_param_combinations > 0 else int(len(self.seqs)/4) # number of [parameter, trajectory, window] combinations we see per epoch
              
    def __getitem__(self, index):
        
        #file = np.random.randint(len(self.seqs))
        file = index
        seqs = self.seqs[file] # choose a file (i.e parameter value) from 1372 possibilies
        
        seq = seqs[0] if self.fixed_ic else np.random.choice(seqs, 1) # choose a random trajectory (i.e IC) within this file from 1 (val) or 4 (train) possibilites

        h5_file = self.h5_files[file]
        h5_path = self.h5_paths[file]
        k, du, dv = self.extract_params_from_path(h5_path)
        
        
        #get data
        try:
            vid = torch.Tensor(np.array(h5_file[f"{seq.item()}/data"], dtype="f")).to(torch.cuda.current_device()) # dim = [101, 128, 128, 2]
        except:
            vid = torch.Tensor(np.array(h5_file[f"{seq}/data"], dtype="f")).to(torch.cuda.current_device())

        #sample a random window from this trajectory starting at 20 to get rid of high residuals (101 - 20 - self.seq_len possibilities)
        start = 20 if self.fixed_window else np.random.randint(20, vid.shape[0] - self.seq_len)
        vid = vid[start:start+self.seq_len].permute((0, 3, 1, 2))
        
        if self.frame_step > 1:
            vid = vid[::self.frame_step]  # only take the video frames
        #return video and parameters
        return vid.reshape(-1, 2, self.image_size, self.image_size), (k, du, dv)



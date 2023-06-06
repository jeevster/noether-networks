import numpy as np
import h5py as h
import random
import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from common import Advection

plt.rcParams['figure.figsize'] = (10, 8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('plotdir', type=str)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--rng_seed', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.plotdir, exist_ok=True)
    plotdir = args.plotdir
    datadir = args.data
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    parameter_combos = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f)) and f.endswith('.hdf5')]
    parameter_combos = random.sample(parameter_combos, args.n)
    for f in tqdm(parameter_combos):
        d = h.File(os.path.join(datadir, f))
        beta = float(d.attrs['beta'])
        data = d['tensor']
        x = np.asarray(d['x-coordinate'])
        t = np.asarray(d['t-coordinate'])
        curr_dir = os.path.join(plotdir, f'beta={beta}')
        os.makedirs(curr_dir, exist_ok=True)
        for seed in tqdm(range(data.shape[0]), leave=False):    
            Advection.plot(data[seed], x, t, beta, seed, curr_dir)
        d.close()

if __name__ == '__main__':
    main()
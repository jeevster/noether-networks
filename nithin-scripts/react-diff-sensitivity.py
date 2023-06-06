from tqdm import tqdm
import os
import random
import h5py as h
import numpy as np
import matplotlib.pyplot as plt
import argparse

from .common import ReactionDiff2D

plt.rcParams['figure.figsize'] = (10, 8)
random.seed(0)
np.random.seed(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, required=True)
    parser.add_argument('plotdir', type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.plotdir, exist_ok=True)
    plotdir = args.plotdir
    datadir = args.data
    parameter_combos = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f)) and f.endswith('.h5')]
    for f in tqdm(parameter_combos):
        d = h.File(os.path.join('./../', f))
        elements = f.split("=")[1:]
        du = float(elements[0].split("_")[0])
        dv = float(elements[1].split("_")[0])
        k = float(elements[2].split(".")[0] + "." + elements[2].split(".")[1])
        seed = '0000'
        data = np.asarray(d[seed]['data'])
        x = np.asarray(d[seed]['grid/x'])
        y = np.asarray(d[seed]['grid/y'])
        t = np.asarray(d[seed]['grid/t'])

        curr_outdir = os.path.join(plotdir, f'k={k}_du={du}_dv={dv}')
        os.makedirs(curr_outdir, exist_ok=True)
        gt_residual, du_residual, dv_residual, k_residual, noise = ReactionDiff2D.sensitivity_analysis(
            data[:, :, :, 0], data[:, :, :, 1], x, y, t, k, du, dv)
        ReactionDiff2D.plot_sensitivity_analysis(gt_residual, du_residual, dv_residual,
            k_residual, noise, k, du, dv, curr_outdir)



if __name__ == '__main__':
    main()
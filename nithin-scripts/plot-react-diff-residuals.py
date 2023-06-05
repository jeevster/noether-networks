import random
import h5py as h
import numpy as np
import os
import argparse

from .common import ReactionDiff2D

random.seed(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, required=True)
    parser.add_argument('plotdir', type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.plotdir, exist_ok=True)
    plotdir = args.plotdir
    datadir = args.data
    parameter_combos = [f for f in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, f)) and f.endswith('.h5')]
    for f in parameter_combos:
        residuals = []
        d = h.File(os.path.join('./../', f))
        elements = f.split("=")[1:]
        du = float(elements[0].split("_")[0])
        dv = float(elements[1].split("_")[0])
        k = float(elements[2].split(".")[0] + "." + elements[2].split(".")[1])

        for seed in d.keys():
            data = np.asarray(d[seed]['data'])
            x = np.asarray(d[seed]['grid/x'])
            y = np.asarray(d[seed]['grid/y'])
            t = np.asarray(d[seed]['grid/t'])
            res = ReactionDiff2D.residual(
                data[:, :, :, 0], data[:, :, :, 1], x, y, t, k, du, dv)
            residuals.append(res)
        residuals = np.asarray(residuals)
        d.close()
        ReactionDiff2D.plot_residuals(residuals, t, x, k, du, dv, plotdir)

if __name__ == '__main__':
    main()
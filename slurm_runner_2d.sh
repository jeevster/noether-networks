#!/bin/bash

EXP=run_data_outer_val=learnable_train=learnable.sh
BASEDIR=config_files/2d_react_diff_copy/ckpt
sbatch $BASEDIR/param_seed=0_slurm/$EXP
sbatch $BASEDIR/param_seed=1_slurm/$EXP
sbatch $BASEDIR/param_seed=2_slurm/$EXP

EXP=run_data_outer_val=learnable_train=learnable_init_frames.sh
BASEDIR=config_files/2d_react_diff_copy/ckpt
sbatch $BASEDIR/param_seed=0_slurm/$EXP
sbatch $BASEDIR/param_seed=1_slurm/$EXP
sbatch $BASEDIR/param_seed=2_slurm/$EXP

EXP=run_data_outer_val=learnable_train=learnable_param_loss.sh
BASEDIR=config_files/2d_react_diff_copy/ckpt
sbatch $BASEDIR/param_seed=0_slurm/$EXP
sbatch $BASEDIR/param_seed=1_slurm/$EXP
sbatch $BASEDIR/param_seed=2_slurm/$EXP

EXP=run_data_outer_val=learnable_train=learnable_param_loss_init_frames.sh
BASEDIR=config_files/2d_react_diff_copy/ckpt
sbatch $BASEDIR/param_seed=0_slurm/$EXP
sbatch $BASEDIR/param_seed=1_slurm/$EXP
sbatch $BASEDIR/param_seed=2_slurm/$EXP

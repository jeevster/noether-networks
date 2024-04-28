#!/bin/bash

# BASEDIR=config_files/1d_diffusion_reaction_multiparam_new/ckpt

# for FILE in $BASEDIR/pinns_sweep_seed=0/*; do
#     echo ${FILE}
#     sbatch ${FILE}
#     sleep 1 # pause to be kind to the scheduler
# done

# for FILE in $BASEDIR/pinns_sweep_seed=1/*; do
#     echo ${FILE}
#     sbatch ${FILE}
#     sleep 1 # pause to be kind to the scheduler
# done

# for FILE in $BASEDIR/pinns_sweep_seed=2/*; do
#     echo ${FILE}
#     sbatch ${FILE}
#     sleep 1 # pause to be kind to the scheduler
# done


# BASEDIR=config_files/1d_burgers_multiparam_new/ckpt

# for FILE in $BASEDIR/pinns_sweep_seed=0/*; do
#     echo ${FILE}
#     sbatch ${FILE}
#     sleep 1 # pause to be kind to the scheduler
# done

# for FILE in $BASEDIR/pinns_sweep_seed=1/*; do
#     echo ${FILE}
#     sbatch ${FILE}
#     sleep 1 # pause to be kind to the scheduler
# done

# for FILE in $BASEDIR/pinns_sweep_seed=2/*; do
#     echo ${FILE}
#     sbatch ${FILE}
#     sleep 1 # pause to be kind to the scheduler
# done

BASEDIR=config_files/1d_advection_multiparam_new/ckpt

for FILE in $BASEDIR/ood_param_seed=0_slurm/*; do
    echo ${FILE}
    sbatch ${FILE}
    sleep 1 # pause to be kind to the scheduler
done

for FILE in $BASEDIR/ood_param_seed=1_slurm/*; do
    echo ${FILE}
    sbatch ${FILE}
    sleep 1 # pause to be kind to the scheduler
done

for FILE in $BASEDIR/ood_param_seed=2_slurm/*; do
    echo ${FILE}
    sbatch ${FILE}
    sleep 1 # pause to be kind to the scheduler
done
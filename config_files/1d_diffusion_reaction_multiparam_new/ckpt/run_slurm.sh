#!/bin/bash
for FILE in seed=0_slurm/*; do
    echo ${FILE}
    sbatch ${FILE}
    sleep 1 # pause to be kind to the scheduler
done

for FILE in seed=1_slurm/*; do
    echo ${FILE}
    sbatch ${FILE}
    sleep 1 # pause to be kind to the scheduler
done

for FILE in /seed=2_slurm/*; do
    echo ${FILE}
    sbatch ${FILE}
    sleep 1 # pause to be kind to the scheduler
done
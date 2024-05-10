#!/bin/bash
#SBATCH -A m4319
#SBATCH -C gpu
#SBATCH -t 5:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH -G 1
#SBATCH --mail-user=divyam123@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular

module load conda
module load nccl/2.17.1-ofi
conda activate /global/cfs/projectdirs/m4319/divyam123/miniconda3/envs/noether_env


DATADIR=$PSCRATCH/burgers_1024_2048_-2_0.3_clean

STOREDIR=/global/cfs/projectdirs/m4319/divyam123/slurm_runs

LOSS=mse
SEED=0
lr=0.00001
LOGDIR=$STOREDIR/slurm_runs_lr_sweep/results_noether_summer/1d_burgers_new_params_clean/ckpt/seed=$SEED/no_norm_steps=5
EMBDIR=$STOREDIR/results_noether_summer/1d_burgers_multiparam_clean/ckpt/seed=$SEED/pretrained_embedding_param_loss/final/best_ckpt_model.pt
python train_noether_net.py \
--use_init_frames_for_params \
--emmbedding_param_loss \
--seed $SEED \
--tailor \
--use_cn \
--use_adam_inner_opt \
--emb_type pde_emb \
--outer_loss_choice mse \
--inner_crit_compare_to pde_zero \
--image_width 128 \
--g_dim 128 \
--z_dim 64 \
--dataset 1d_burgers_multiparam \
--data_root $DATADIR \
--num_trials 1 \
--n_past 2 \
--n_future 2 \
--num_threads 0 \
--inner_crit_mode mse \
--enc_dec_type vgg \
--num_epochs_per_val 1 \
--fno_modes 16 \
--fno_width 64 \
--fno_layers 3 \
--emb_dim 64 \
--val_batch_size 1 \
--train_batch_size 16 \
--train_num_inner_steps 1 \
--val_num_inner_steps 5 \
--num_jump_steps 0 \
--n_epochs 200 \
--inner_lr 0.00001 \
--val_inner_lr 0.00001 \
--outer_lr 0.0001 \
--outer_opt_model_weights \
--random_weights \
--only_twenty_degree \
--frame_step 1 \
--center_crop 1080 \
--num_emb_frames 2 \
--horiz_flip \
--reuse_lstm_eps \
--num_learned_parameters 1 \
--use_partials \
--save_checkpoint \
--ckpt_outer_loss \
--warmstart_emb_path $EMBDIR \
--log_dir $LOGDIR/run_data_outer_train=learnable_val=learnable_param_loss_init_frames_sgd_lr=$lr/ \
--channels 1 \
--random_weights \
--batch_norm_to_group_norm \
--model_path ./checkpoints/pdes/t_past2/batch_5d

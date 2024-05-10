#!/bin/bash
#SBATCH -A m4319
#SBATCH -C gpu
#SBATCH -t 01:00:00
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


DATADIR=$PSCRATCH/advection_log_space_res_1024_start=1e-2_end=2_multiparam_100

STOREDIR=/global/cfs/projectdirs/m4319/divyam123/slurm_runs

SEED=0
LOSS=mse
DATAREG=1
PDEREG=1
python train_noether_net_checkpointing_non_meta_2.py \
--seed $SEED \
--data_loss_reg $DATAREG \
--pde_loss_reg $PDEREG \
--pinn_outer_loss \
--inner_opt_all_model_weights \
--use_adam_inner_opt \
--emb_type pde_const_emb \
--relative_data_loss \
--outer_loss_choice mse \
--inner_crit_compare_to pde_zero \
--image_width 128 \
--g_dim 128 \
--z_dim 64 \
--dataset 1d_advection_multiparam \
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
--val_num_inner_steps 1 \
--num_jump_steps 0 \
--n_epochs 200 \
--inner_lr 0.0001 \
--val_inner_lr 0.0001 \
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
--log_dir  $STOREDIR/pinns_sweep/1d_advection_new_params/ckpt/seed=$SEED/param_no_norm_steps=5/run_pinns_data=${DATAREG}_pde=${PDEREG}/ \
--warmstart_emb_path $STOREDIR/results_noether_summer/1d_advection_new_params/pre_trained_embeddings/seed=$SEED/pretrained_embedding_param_loss/final/best_ckpt_model.pt \
--channels 1 \
--random_weights \
--batch_norm_to_group_norm \
--model_path ./checkpoints/pdes/t_past2/batch_5d \

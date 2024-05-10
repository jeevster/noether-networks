#!/bin/bash
#SBATCH -A m4319
#SBATCH -C gpu
#SBATCH -t 23:59:59
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --gpu-bind=none
#SBATCH -G 4
#SBATCH --mail-user=divyam123@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -o /global/cfs/projectdirs/m4319/divyam123/run4_none.out

module load conda
module load nccl/2.17.1-ofi
conda activate /global/cfs/projectdirs/m4319/divyam123/miniconda3/envs/noether_env

python train_noether_net_checkpointing_non_meta_2_distributed.py \
--tailor \
--pinn_outer_loss \
--use_cn \
--use_adam_inner_opt \
--emb_type pde_const_emb \
--relative_data_loss \
--outer_loss_choice mse \
--inner_crit_compare_to pde_zero \
--image_width 128 \
--g_dim 128 \
--z_dim 64 \
--dataset 2d_reacdiff_multiparam \
--data_root $PSCRATCH/ReactionDiffusion \
--num_trials 1 \
--n_past 2 \
--n_future 2 \
--num_threads 0 \
--inner_crit_mode mse \
--enc_dec_type vgg \
--num_epochs_per_val 1 \
--fno_modes 16 \
--fno_width 256 \
--fno_layers 4 \
--emb_dim 64 \
--val_batch_size 1 \
--train_batch_size 8 \
--train_num_inner_steps 1 \
--val_num_inner_steps 5 \
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
--reload_latest \
--restore_path /global/cfs/projectdirs/m4319/divyam123/2d_react_diff_multiparam/TRIAL/no_norm_steps=5/data_true_outer_true_inner_num_steps=5_no_norm_opt=adam_weights=cn/Sat-Feb--3-05.47.35-2024_past=2_future=2_tailor=PDE_Const_1_PINN_Outer_Loss_Weight=1.0 \
--restore_epoch 191 \
--save_checkpoint \
--log_dir /global/cfs/projectdirs/m4319/divyam123/2d_react_diff_multiparam/TRIAL/no_norm_steps=5/data_true_outer_true_inner_num_steps=5_no_norm_opt=adam_weights=cn/ \
--warmstart_emb_path best_ckpt_model.pt \
--channels 2 \
--random_weights \
--batch_norm_to_group_norm \
--model_path ./checkpoints/pdes/t_past2/batch_5d \

# --norm no_norm \ --reload_latest \
#!/bin/bash
DATADIR=/data/divyam123/react_diff_4096_8192_nu=1e-2_1e-1_rho=1_10
# DATADIR=/data/nithinc/pdebench/new_parameters/2D/ReactionDiffusion
#changes relative to vanilla Noether: --inner_opt_all_model_weights, no prior/posterior so no kl loss
# python train_noether_net_checkpointing_non_meta_2.py \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \ 2d_reacdiff_multiparam \
# --data_root $DATADIR \
# --num_trials 1 \
# --n_past 2 \
# --n_future 2 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to prev \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 4 \
# --emb_dim 64 \
# --batch_size 8 \
# --train_num_inner_steps 1 \
# --num_jump_steps 0 \
# --n_epochs 200 \
# --inner_lr .0001 \
# --val_inner_lr .0001 \
# --outer_lr .0001 \
# --outer_opt_model_weights \
# --random_weights \
# --only_twenty_degree \
# --frame_step 1 \
# --center_crop 1080 \
# --num_emb_frames 2 \
# --horiz_flip \
# --reuse_lstm_eps \
# --num_learned_parameters 1 \
# --use_partials \
# --log_dir ./results_summer_new/advection/2d_settings/no_pretraining \
# --channels 1 \--ckpt_every 10 \
# --tailor \
# --random_weights \--ckpt_outer_loss \
# --ckpt_inner_loss \
# --inner_opt_all_model_weights \
# --batch_norm_to_group_norm \
# --advection_emb \
# --save_checkpoint \
# --no_data_loss \--save_checkpoint \
# --use_embedding \ --inner_opt_all_model_weights \ 
# --pinn_outer_loss \ use_true_params_train\ --use_adam_inner_opt \ --use_cn \
# --teacher_forcing \ --param_loss \ inner_opt_all_model_weights \--norm no_norm \

SEED=1
LOSS=mse
python train_noether_net_checkpointing_pino_final_metrics.py \
--seed $SEED \
--operator_loss \
--pinn_outer_loss \
--inner_opt_all_model_weights \
--use_adam_inner_opt \
--outer_loss_choice mse \
--relative_data_loss \
--num_tuning_steps 5 \
--emb_type pde_const_emb \
--image_width 128 \
--g_dim 128 \
--z_dim 64 \
--dataset 1d_diffusion_reaction_multiparam \
--data_root $DATADIR \
--num_trials 1 \
--n_past 2 \
--n_future 2 \
--num_threads 0 \
--inner_crit_mode mse \
--inner_crit_compare_to pde_log \
--enc_dec_type vgg \
--num_epochs_per_val 1 \
--fno_modes 16 \
--fno_width 64 \
--fno_layers 3 \
--emb_dim 64 \
--val_batch_size 1 \
--train_batch_size 16 \
--train_num_inner_steps 1 \
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
--log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction_multiparam_new/slurm_baseline_4k/seed=$SEED/param_no_norm_steps=5/run_pino_fixed \
--reload_dir /data/divyam123/slurm_runs/results_noether_summer/1d_diffusion_reaction_multiparam_new/ckpt/seed=$SEED/param_no_norm_steps=5/run_pino_fixed/final/ckpt_model.pt \
--channels 1 \
--random_weights \
--batch_norm_to_group_norm \
--model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_pino_manual.py \
# --operator_loss \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --outer_loss_choice mse \
# --relative_data_loss \
# --num_tuning_steps 5 \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 1 \
# --n_past 2 \
# --n_future 2 \
# --num_threads 0 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --val_batch_size 1 \
# --train_batch_size 16 \
# --train_num_inner_steps 1 \
# --num_jump_steps 0 \
# --n_epochs 200 \
# --inner_lr 0.0001 \
# --val_inner_lr 0.0001 \
# --outer_lr 0.0001 \
# --outer_opt_model_weights \
# --random_weights \
# --only_twenty_degree \
# --frame_step 1 \
# --center_crop 1080 \
# --num_emb_frames 2 \
# --horiz_flip \
# --reuse_lstm_eps \
# --num_learned_parameters 1 \
# --use_partials \
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_REACTION/TRIAL/no_norm_steps=5/PINO \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \
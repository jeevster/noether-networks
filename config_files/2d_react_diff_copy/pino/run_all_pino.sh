#!/bin/bash
DATADIR=/data/nithinc/pdebench/new_parameters/2D/ReactionDiffusion
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
# --teacher_forcing \ --param_loss \ inner_opt_all_model_weights \--norm layer_norm \

# python train_noether_net_checkpointing_pino_manual.py \
# --norm layer_norm \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --outer_loss_choice mse \
# --relative_data_loss \
# --num_tuning_steps 1 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/non_meta_baseline/PINO_SGD_V_ADAM/SGD_5_steps_layer_norm \
# --warmstart_emb_path best_ckpt_model_advection.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_pino_manual.py \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --outer_loss_choice mse \
# --relative_data_loss \
# --num_tuning_steps 1 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/non_meta_baseline/PINO_SGD_V_ADAM/SGD_5_steps_no_norm \
# --warmstart_emb_path best_ckpt_model_advection.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \
#####################################################################################################################################
# python train_noether_net_checkpointing_pino_manual.py \
# --operator_loss \
# --norm layer_norm \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --outer_loss_choice mse \
# --relative_data_loss \
# --num_tuning_steps 1 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/PINO/ADAM_1_steps_layer_norm \
# --warmstart_emb_path best_ckpt_model_advection.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_pino_manual.py \
# --operator_loss \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --outer_loss_choice mse \
# --relative_data_loss \
# --num_tuning_steps 1 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/PINO/ADAM_1_steps_no_norm \
# --warmstart_emb_path best_ckpt_model_advection.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_pino_manual.py \
# --operator_loss \
# --norm layer_norm \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/PINO/ADAM_5_steps_layer_norm \
# --warmstart_emb_path best_ckpt_model_advection.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/PINO/ADAM_5_steps_no_norm \
# --warmstart_emb_path best_ckpt_model_advection.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \
# ################################################################################################################################################
# python train_noether_net_checkpointing_non_meta_2.py \
# --pinn_outer_loss \
# --conditioning \
# --teacher_forcing \
# --use_embedding \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 2 \
# --n_future 10 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/teacher_norm=None_past=2_fut=10 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --norm layer_norm \
# --pinn_outer_loss \
# --teacher_forcing \
# --conditioning \
# --use_embedding \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 2 \
# --n_future 10 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/teacher_norm=layer_past=2_fut=10 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --norm layer_norm \
# --pinn_outer_loss \
# --conditioning \
# --teacher_forcing \
# --use_embedding \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 1 \
# --n_future 10 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/teacher_norm=layer_norm_past=1_fut=10 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --pinn_outer_loss \
# --conditioning \
# --teacher_forcing \
# --use_embedding \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 1 \
# --n_past 1 \
# --n_future 10 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/teacher_norm=None_past=1_fut=10 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --pinn_outer_loss \
# --conditioning \
# --teacher_forcing \
# --use_embedding \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 2 \
# --n_future 2 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/teacher_norm=None_past=2_fut=2 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --norm layer_norm \
# --pinn_outer_loss \
# --teacher_forcing \
# --conditioning \
# --use_embedding \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 2 \
# --n_future 2 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/teacher_norm=layer_past=2_fut=2 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \
# ################################################################################################################################################
# python train_noether_net_checkpointing_non_meta_2.py \
# --pinn_outer_loss \
# --conditioning \
# --use_embedding \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 2 \
# --n_future 10 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/learned_norm=None_past=2_fut=10 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --norm layer_norm \
# --pinn_outer_loss \
# --conditioning \
# --use_embedding \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 2 \
# --n_future 10 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/learned_norm=layer_norm_past=2_fut=10 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --pinn_outer_loss \
# --conditioning \
# --use_embedding \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 1 \
# --n_past 1 \
# --n_future 10 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/learned_norm=None_past=1_fut=10 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --norm layer_norm \
# --pinn_outer_loss \
# --conditioning \
# --use_embedding \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 1 \
# --n_future 10 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/learned_norm=layer_norm_past=1_fut=10 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --pinn_outer_loss \
# --conditioning \
# --use_embedding \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 2 \
# --n_future 2 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/learned_norm=None_past=2_fut=2 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --norm layer_norm \
# --pinn_outer_loss \
# --conditioning \
# --use_embedding \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 2 \
# --n_future 2 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/learned_norm=layer_norm_past=2_fut=2 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \
# ################################################################################################################################################
# python train_noether_net_checkpointing_non_meta_2.py \
# --conditioning \
# --pinn_outer_loss \
# --use_true_params_train \
# --use_true_params_val \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 2 \
# --n_future 10 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/true_norm=None_past=2_fut=10 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --norm layer_norm \
# --conditioning \
# --pinn_outer_loss \
# --use_true_params_train \
# --use_true_params_val \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 2 \
# --n_future 10 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/true_norm=layer_norm_past=2_fut=10 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --norm layer_norm \
# --conditioning \
# --pinn_outer_loss \
# --use_true_params_train \
# --use_true_params_val \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 1 \
# --n_future 10 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/true_norm=layer_norm_past=1_fut=10 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --conditioning \
# --pinn_outer_loss \
# --use_true_params_train \
# --use_true_params_val \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 1 \
# --n_past 1 \
# --n_future 10 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/true_norm=None_past=1_fut=10 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --conditioning \
# --pinn_outer_loss \
# --use_true_params_train \
# --use_true_params_val \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 2 \
# --n_future 2s \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/true_norm=None_past=2_fut=2 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --norm layer_norm \
# --conditioning \
# --pinn_outer_loss \
# --use_true_params_train \
# --use_true_params_val \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 2 \
# --n_future 2 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/true_norm=layer_norm_past=2_fut=2 \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

################################################################################################################################################
# python train_noether_net_checkpointing_non_meta_2.py \
# --norm layer_norm \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --emb_type pde_const_emb \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 1 \
# --n_past 2 \
# --n_future 10 \
# --num_threads 0 \
# --inner_crit_mode mse \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/PINNS_layer_norm_past=2_fut=10/ \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --norm layer_norm \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --emb_type pde_const_emb \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 1 \
# --n_past 1 \
# --n_future 10 \
# --num_threads 0 \
# --inner_crit_mode mse \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/PINNS_layer_norm_past=1_fut=10/ \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --emb_type pde_const_emb \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 1 \
# --n_past 2 \
# --n_future 10 \
# --num_threads 0 \
# --inner_crit_mode mse \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/PINNS_no_norm_past=2_fut=10/ \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --emb_type pde_const_emb \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_diffusion_reaction_multiparam \
# --data_root $DATADIR \
# --num_trials 1 \
# --n_past 1 \
# --n_future 10 \
# --num_threads 0 \
# --inner_crit_mode mse \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/PINNS_no_norm_past=1_fut=10/ \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --emb_type pde_const_emb \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
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
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/PINNS_no_norm_past=2_fut=2/ \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --norm layer_norm \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --emb_type pde_const_emb \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
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
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/explicit_conditioning/PINNS_layer_norm_past=2_fut=2/ \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \
######################################################################################################################################
# python train_noether_net_checkpointing_non_meta_2.py \
# --norm layer_norm \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --emb_type pde_const_emb \
# --outer_loss_choice old \
# --inner_crit_compare_to pde_log \
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
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/explicit_conditioning/old_PINNS_layer_norm_past=2_fut=2/ \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --norm instance_norm \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --emb_type pde_const_emb \
# --outer_loss_choice old \
# --inner_crit_compare_to pde_log \
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
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/explicit_conditioning/old_PINNS_instance_norm_past=2_fut=2/ \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --norm batch_norm \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --emb_type pde_const_emb \
# --outer_loss_choice old \
# --inner_crit_compare_to pde_log \
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
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/explicit_conditioning/old_PINNS_batch_norm_past=2_fut=2/ \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2.py \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --emb_type pde_const_emb \
# --outer_loss_choice old \
# --inner_crit_compare_to pde_log \
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
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --batch_size 16 \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/explicit_conditioning/old_PINNS_no_norm_past=2_fut=2/ \
# --warmstart_emb_path best_1d_diff_react_multiparam.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \
######################################################################################################################################################
######################################################################################################################################################
# python train_noether_net_checkpointing_pino_manual.py \
# --operator_loss \
# --norm layer_norm \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_cn \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/PINO/ADAM_STEPS=5_CN_DYNAMICS_TAILORING_LAYER_NORM \
# --warmstart_emb_path best_ckpt_model_advection.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_pino_manual.py \
# --operator_loss \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_cn \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/PINO/ADAM_STEPS=5_CN_DYNAMICS_TAILORING_NO_NORM \
# --warmstart_emb_path best_ckpt_model_advection.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \
# # --inner_opt_all_model_weights \
# python train_noether_net_checkpointing_pino_manual.py \
# --operator_loss \
# --norm layer_norm \
# --pinn_outer_loss \
# --use_cn \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/PINO/ADAM_STEPS=5_CN_TAILORING_LAYER_NORM \
# --warmstart_emb_path best_ckpt_model_advection.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \
# # --inner_opt_all_model_weights \
# python train_noether_net_checkpointing_pino_manual.py \
# --operator_loss \
# --pinn_outer_loss \
# --use_cn \
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
# --log_dir /data/divyam123/results_noether_summer/1d_diffusion_reaction/trial/comparisions_2/PINO/ADAM_STEPS=5_CN_TAILORING_NO_NORM \
# --warmstart_emb_path best_ckpt_model_advection.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

#!/bin/bash
DATADIR=/data/nithinc/pdebench/new_parameters/2D/ReactionDiffusion
#task1 (pure data loss): --emb_type pde_const_emb 3:18
#task2 (pinn): --emb_type pde_const_emb  before 322
#              --pinn_outer_loss \
#task3 (true pde loss): --no_data_loss 324
#                       --emb_type pde_const_emb \
#                       --pinn_outer_loss \
#task4 (data, learned): 1115
#                        --tailor \
#                        --emb_type pde_emb \
#task5 (data,true ): 1116 \\ 145
#                       --tailor \
#                       --emb_type pde_const_emb
#task6 (data true, learned ): 1427 \\ 143
#                       --tailor \
#                       --pinn_outer_loss \
#                       --emb_type pde_emb \
#task7 (data learned, learned): 1428
#                       --tailor \
#                       --pinn_outer_loss \
#                       --learned_pinn_loss \
#                       --emb_type pde_emb \
#task8 (data true, true ):  2059 or 9? \\ 142
#                       --tailor \
#                       --pinn_outer_loss \
#                       --emb_type pde_const_emb \
#task9 (true, learned): 901 \\ 140
#                       --tailor \
#                       --pinn_outer_loss \
#                       --no_data_loss \
#                       --emb_type pde_emb \
#'1d_burgers_multiparam':
#'1d_advection_multiparam':
#'1d_diffusion_reaction_multiparam':
#'2d_reacdiff_multiparam':
# --warmstart_emb_path best_ckpt_model.pt \
# --pinn_outer_loss \

python train_noether_net_checkpointing_non_meta_2.py \
--pinn_outer_loss \
--emb_type pde_const_emb \
--image_width 128 \
--g_dim 128 \
--z_dim 64 \
--dataset 2d_reacdiff_multiparam \
--data_root $DATADIR \
--num_trials 1 \
--n_past 2 \
--n_future 2 \
--num_threads 0 \
--ckpt_every 10 \
--inner_crit_mode mse \
--inner_crit_compare_to pde_log \
--enc_dec_type vgg \
--num_epochs_per_val 1 \
--fno_modes 16 \
--fno_width 256 \
--fno_layers 4 \
--emb_dim 64 \
--batch_size 8 \
--num_inner_steps 1 \
--num_jump_steps 0 \
--n_epochs 200 \
--inner_lr .0001 \
--val_inner_lr .0001 \
--outer_lr .0001 \
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
--ckpt_inner_loss \
--log_dir /data/divyam123/results_noether_summer/2d_react_diff/pretrained_embedding_PINO_manual/ \
--warmstart_emb_path best_ckpt_model.pt \
--channels 2 \
--random_weights \
--inner_opt_all_model_weights \
--batch_norm_to_group_norm \
--model_path ./checkpoints/pdes/t_past2/batch_5d
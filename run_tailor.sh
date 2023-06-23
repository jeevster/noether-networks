#!/bin/bash
 
DATADIR=/data/nithinc/pdebench/new_parameters/2D/ReactionDiffusion

#changes relative to vanilla Noether: --inner_opt_all_model_weights, no prior/posterior so no kl loss
python train_noether_net_checkpointing.py \
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
--tailor \
--inner_crit_mode mse \
--inner_crit_compare_to pde_log \
--enc_dec_type vgg \
--emb_type pde_emb \
--num_epochs_per_val 1 \
--fno_modes 16 \
--fno_width 256 \
--fno_layers 4 \
--emb_dim 64 \
--batch_size 8 \
--num_inner_steps 1 \
--num_jump_steps 0 \
--n_epochs 500 \
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
--warmstart_emb_path /home/sanjeevr/noether-networks/results_summer/pderesidualloss_onlylearnk_lr1e-3_bs16_use_partials_fixedpderesidual_2embframes/Mon-Jun-12-17.15.29-2023_past=2_future=2_tailor=PDE/best_ckpt_model.pt \
--log_dir ./results_noether_summer/TEST/ \
--channels 2 \
--random_weights \
--inner_opt_all_model_weights \
--batch_norm_to_group_norm \
--model_path ./checkpoints/pdes/t_past2/batch_5d


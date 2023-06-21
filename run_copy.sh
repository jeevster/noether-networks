#!/bin/bash
 
DATADIR=/data/divyam123/fenics_fixed

#changes relative to vanilla Noether: --inner_opt_all_model_weights, no prior/posterior so no kl loss
python train_embedding.py \
--image_width 128 \
--g_dim 128 \
--z_dim 64 \
--dataset 1d_burgers_multiparam \
--data_root $DATADIR \
--num_trials 1 \
--n_past 2 \
--n_future 2 \
--num_threads 0 \
--ckpt_every 10 \
--inner_crit_mode mse \
--inner_crit_compare_to prev \
--enc_dec_type vgg \
--emb_type conserved \
--num_epochs_per_val 1 \
--fno_modes 16 \
--fno_width 64 \
--fno_layers 3 \
--emb_dim 64 \
--batch_size 16 \
--num_inner_steps 1 \
--num_jump_steps 0 \
--n_epochs 200 \
--inner_lr .0001 \
--val_inner_lr .0001 \
--outer_lr .001 \
--outer_opt_model_weights \
--random_weights \
--only_twenty_degree \
--frame_step 1 \
--center_crop 1080 \
--num_emb_frames 150 \
--horiz_flip \
--reuse_lstm_eps \
--num_learned_parameters 1 \
--use_partials \
--log_dir ./results_summer_new/burgers/ \
--channels 1 \
--tailor \
--random_weights \
--inner_opt_all_model_weights \
--batch_norm_to_group_norm \
--burgers \

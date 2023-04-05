#!/bin/bash
 
DATADIR=/data/nithinc/generated_pdebench/2D/ReactionDiffusion

#changes relative to vanilla Noether: --inner_opt_all_model_weights, no prior/posterior so no kl loss
python train_embedding.py \
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
--inner_crit_compare_to prev \
--enc_dec_type vgg \
--emb_type conserved \
--num_epochs_per_val 1 \
--fno_modes 16 \
--fno_width 128 \
--fno_layers 2 \
--emb_dim 64 \
--pde_emb \
--batch_size 10 \
--num_inner_steps 1 \
--num_jump_steps 0 \
--n_epochs 100 \
--train_set_length 1000 \
--test_set_length 200 \
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
--log_dir ./results/multiparam_test/ \
--channels 2 \
--tailor \
--random_weights \
--inner_opt_all_model_weights \
--batch_norm_to_group_norm \
--model_path ./checkpoints/pdes/t_past2/batch_5d

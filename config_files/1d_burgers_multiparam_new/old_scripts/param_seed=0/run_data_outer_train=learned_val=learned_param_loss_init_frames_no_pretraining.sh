#!/bin/bash
DATADIR=/data/divyam123/burgers_1024_2048_-2_0.3_clean
SEED=0
LOGDIR=/data/divyam123/slurm_runs/results_noether_summer/1d_burgers_new_params_clean/ckpt/seed=$SEED/param_no_norm_steps=5
EMBDIR=/data/divyam123/slurm_runs/results_noether_summer/1d_burgers_multiparam_clean/ckpt/seed=$SEED/pretrained_embedding_param_loss/final/best_ckpt_model.pt

LOSS=mse
python train_noether_net_checkpointing_non_meta_2.py \
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
--log_dir $LOGDIR/run_data_outer_train=learned_val=learned_param_loss_init_frames_no_pretraining/ \
--channels 1 \
--random_weights \
--batch_norm_to_group_norm \
--model_path ./checkpoints/pdes/t_past2/batch_5d \

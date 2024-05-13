#!/bin/bash
DATADIR=/data/divyam123/burgers_4096_8192_-2_0.3
for SEED in 0 1 2
do
    LOGDIR=/data/divyam123/slurm_runs_all/1d_burgers_new_params_clean/baseline/seed=$SEED/no_norm_steps=5
    RELOADDIR=/data/divyam123/slurm_runs_all/1d_burgers_new_params_clean/ckpt/seed=$SEED/no_norm_steps=5
    python train_noether_net_final_inference.py \
    --percent_train 0.0 \
    --seed $SEED \
    --conditioning \
    --single_field \
    --param_scale_shift_conditioning \
    --pinn_outer_loss \
    --use_true_params_train \
    --use_true_params_val \
    --inner_opt_all_model_weights \
    --use_adam_inner_opt \
    --outer_loss_choice mse \
    --inner_crit_compare_to pde_zero \
    --emb_type pde_const_emb \
    --image_width 128 \
    --g_dim 128 \
    --z_dim 64 \
    --dataset 1d_burgers_multiparam \
    --data_root $DATADIR \
    --num_trials 2 \
    --n_past 2 \
    --n_future 2 \
    --num_threads 0 \
    --ckpt_every 10 \
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
    --log_dir $LOGDIR/run_true_conditioning_single_scale_shift \
    --reload_dir $RELOADDIR/run_true_conditioning_single_scale_shift/final/best_outer_val_ckpt_model.pt \
    --channels 1 \
    --random_weights \
    --batch_norm_to_group_norm \
    --model_path ./checkpoints/pdes/t_past2/batch_5d
done

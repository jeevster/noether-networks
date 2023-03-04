#!/bin/bash

mamba activate noether

echo "n_future=$1"
echo "batch_size=$2" 

python train_noether_net.py --image_width 128 --g_dim 128 --z_dim 64 --dataset 2d_reacdiff --data_root /data/nithinc/PDEs/2D/diffusion-reaction --num_trials 1 --n_past $1 --n_future 2 --num_threads 0 --ckpt_every 10 --inner_crit_mode mse --enc_dec_type vgg --emb_type conserved --num_epochs_per_val 1 --emb_dim 64 --conv_emb --batch_size $2 --num_inner_steps 1 --num_jump_steps 0 --n_epochs 100 --train_set_length 100 --test_set_length 20 --inner_lr .0001 --val_inner_lr .0001 --outer_lr .0001 --outer_opt_model_weights --random_weights --only_twenty_degree --frame_step 1 --center_crop 1080 --num_emb_frames 2 --horiz_flip --batch_norm_to_group_norm --reuse_lstm_eps --log_dir ./results/2d_reacdiff_convemb/past2_fut$1_train100_val20_lr0.0001_bs5_tailor/ --channels 2 --tailor --random_weights --model_path ./checkpoints/pdes/t_past$1/batch_$2

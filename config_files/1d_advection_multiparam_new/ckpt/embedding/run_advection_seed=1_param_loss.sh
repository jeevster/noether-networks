# DATADIR=/data/divyam123/burgers_1024_2048_-2_0.3
DATADIR=/data/divyam123/advection_log_space_res_1024_start=1e-2_end=2_multiparam_100
# DATADIR=/data/divyam123/react_diff_1024_2048_nu=1e-2_1e-1_rho=1_10
# 1d_advection_multiparam, 1d_diffusion_reaction_multiparam, 1d_burgers_multiparam
LOSS=mse
SEED=1
# --save_checkpoint \
python train_embedding.py \
--seed $SEED \
--param_loss \
--gradient_clipping \
--save_checkpoint \
--outer_loss $LOSS \
--image_width 128 \
--g_dim 128 \
--z_dim 64 \
--advection_emb \
--dataset 1d_advection_multiparam \
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
--pde_emb \
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
--log_dir /data/divyam123/slurm_runs/results_noether_summer/1d_advection_new_params/pre_trained_embeddings/seed=$SEED/pretrained_embedding_param_loss \
--channels 1 \
--tailor \
--random_weights \
--inner_opt_all_model_weights \
--batch_norm_to_group_norm \

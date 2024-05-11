DATADIR=/data/nithinc/pdebench/new_parameters/2D/ReactionDiffusion

LOSS=mse
# --save_checkpoint \
SEED=0
python train_embedding.py \
--param_loss \
--save_checkpoint \
--outer_loss $LOSS \
--image_width 128 \
--g_dim 128 \
--z_dim 64 \
--advection_emb \
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
--fno_width 256 \
--fno_layers 4 \
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
--log_dir /data/divyam123/slurm_runs/results_noether_summer/2d_react_diff/embeddings/pre_trained_embedding_param_loss_seed=$SEED \
--channels 2 \
--tailor \
--random_weights \
--inner_opt_all_model_weights \
--batch_norm_to_group_norm \
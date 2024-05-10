DATADIR=/data/nithinc/pdebench/new_parameters/2D/ReactionDiffusion


python train_noether_net_checkpointing_non_meta_2_final_metrics.py \
--fixed_ic \
--fixed_window \
--conditioning \
--pinn_outer_loss \
--use_true_params_train \
--use_true_params_val \
--inner_opt_all_model_weights \
--use_adam_inner_opt \
--emb_type pde_const_emb \
--relative_data_loss \
--outer_loss_choice mse \
--inner_crit_compare_to pde_zero \
--image_width 128 \
--g_dim 128 \
--z_dim 64 \
--dataset 2d_reacdiff_multiparam \
--data_root $DATADIR \
--num_trials 1 \
--n_past 2 \
--n_future 50 \
--num_threads 0 \
--inner_crit_mode mse \
--enc_dec_type vgg \
--num_epochs_per_val 1 \
--fno_modes 16 \
--fno_width 256 \
--fno_layers 4 \
--emb_dim 64 \
--val_batch_size 1 \
--train_batch_size 4 \
--val_num_inner_steps 1 \
--train_num_inner_steps 1 \
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
--reload_dir /data/divyam123/results_noether_summer/2d_react_diff_multiparam/2d_ckpt/true_conditioning/Sun-Feb-25-19.29.25-2024_past=2_future=2_tailor=None_PINN_Outer_Loss_Weight=1.0 \
--log_dir /data/divyam123/results_noether_summer/2d_react_diff_multiparam/baseline_1024_fixed/no_norm_steps=5/true_conditioning/ \
--channels 2 \
--random_weights \
--batch_norm_to_group_norm \
--model_path ./checkpoints/pdes/t_past2/batch_5d \

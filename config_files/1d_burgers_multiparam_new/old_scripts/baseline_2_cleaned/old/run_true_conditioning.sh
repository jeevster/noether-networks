DATADIR=/data/divyam123/burgers_1024_2048_-2_0.3_clean

# --relative_data_loss \

python train_noether_net_checkpointing_non_meta_2_final_metrics.py \
--conditioning \
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
--reload_dir /data/divyam123/results_noether_summer/1d_burgers_new_params_clean/ckpt/no_norm_steps=5/mse_explicit_conditioning/true_norm=no_norm_past=2_fut=2/Thu-Mar--7-12.23.20-2024_past=2_future=2_tailor=None_PINN_Outer_Loss_Weight=1.0 \
--log_dir /data/divyam123/results_noether_summer/1d_burgers_new_params_clean/baseline_2_random/no_norm_steps=5/mse_explicit_conditioning/true_norm=no_norm_past=2_fut=2/Mon-Feb-26-06.49.07-2024_past=2_future=2_tailor=None_PINN_Outer_Loss_Weight=1.0 \
--channels 1 \
--random_weights \
--batch_norm_to_group_norm \
--model_path ./checkpoints/pdes/t_past2/batch_5d \

# python train_noether_net_checkpointing_non_meta_2_final_metrics.py \
# --fixed_ic \
# --fixed_window \
# --conditioning \
# --pinn_outer_loss \
# --use_true_params_train \
# --use_true_params_val \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --emb_type pde_const_emb \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_burgers_multiparam \
# --data_root $DATADIR \
# --num_trials 2 \
# --n_past 2 \
# --n_future 5 \
# --num_threads 0 \
# --ckpt_every 10 \
# --inner_crit_mode mse \
# --inner_crit_compare_to pde_log \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --val_batch_size 1 \
# --train_batch_size 16 \
# --train_num_inner_steps 1 \
# --val_num_inner_steps 1 \
# --num_jump_steps 0 \
# --n_epochs 200 \
# --inner_lr 0.0001 \
# --val_inner_lr 0.0001 \
# --outer_lr 0.0001 \
# --outer_opt_model_weights \
# --random_weights \
# --only_twenty_degree \
# --frame_step 1 \
# --center_crop 1080 \
# --num_emb_frames 2 \
# --horiz_flip \
# --reuse_lstm_eps \
# --num_learned_parameters 1 \
# --use_partials \
# --reload_dir /data/divyam123/results_noether_summer/1d_burgers_new_params/ckpt/no_norm_steps=5/mse_explicit_conditioning/true_norm=no_norm_past=2_fut=2/Mon-Feb-26-06.49.07-2024_past=2_future=2_tailor=None_PINN_Outer_Loss_Weight=1.0 \
# --log_dir /data/divyam123/results_noether_summer/1d_burgers_new_params/baseline_5/no_norm_steps=5/mse_explicit_conditioning/true_norm=no_norm_past=2_fut=2 \
# --warmstart_emb_path best_ckpt_burgers_multiparam_mse.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \
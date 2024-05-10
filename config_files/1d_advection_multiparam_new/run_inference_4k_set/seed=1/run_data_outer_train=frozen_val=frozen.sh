DATADIR=/data/divyam123/advection_subsample
#task1 (pure data loss): --emb_type pde_const_emb  0.00045387641964250226
#task2 (pinn): --emb_type pde_const_emb  2343 -- batch_size: 1 outer_lr: 0.000944657547300611
#              --pinn_outer_loss \
#task3 (true pde loss): --no_data_loss 2345
#                       --emb_type pde_const_emb \
#                       --pinn_outer_loss \
#task4 (data, learned): 135 -- inner_lr: 0.00010052140946737732, outer_lr: 0.0005396503223183395, val_inner_lr: 0.00012866431426295137 batch_size: 4
#                        --tailor \
#                        --emb_type pde_emb \
#task5 (data,true ): 146 || 700 -- 
#                       --tailor \
#                       --emb_type pde_const_emb
#task6 (data true, learned ):  150 || 722 inner_lr: 0.0008611894898233637, outer_lr: 0.0006045857380020396, val_inner_lr: 0.0009519369616581873, batch_size: 4
#                       --tailor \
#                       --pinn_outer_loss \
#                       --emb_type pde_emb \
#task7 (data learned, learned ): 151 --   val_ir = 2.6969134166141357e-05, outer_lr = 6.927521235814095e-05, inner_lr = 5.493274723048728e-05, bs = 1
#                       --tailor \
#                       --pinn_outer_loss \
#                       --learned_pinn_loss \
#                       --emb_type pde_emb \
#task8 (data true, true ): 152 || 647 -- inner_lr: 3.39517034395788e-05, outer_lr: 6.25091343382013e-05, val_inner_lr: 1.755095307425924e-05, batch_size: 1
#                       --tailor \
#                       --pinn_outer_loss \
#                       --emb_type pde_const_emb \
#task9 (true, learned): 153 || 633 -- inner_lr: 1.1087405439921263e-05, outer_lr: 0.0029598016020391895, val_inner_lr: 0.00014047475304588118
#                       --tailor \
#                       --pinn_outer_loss \
#                       --no_data_loss \
#                       --emb_type pde_emb \
#'1d_advection_multiparam':
#'1d_advection_multiparam':
#'1d_advection_multiparam':
#'2d_reacdiff_multiparam':
#--warmstart_emb_path best_ckpt_model_advection.pt \
#--add_general_learnable \ 1 inner step- 1258 | 5 1332 --inner_opt_all_model_weights \
#--use_adam_inner_opt
#--norm instance_norm \
#--use_cn \
# --save_checkpoint \
# --ckpt_outer_loss \
# --ckpt_inner_loss \
# --ckpt_every 10 \--relative_data_loss \
# python train_noether_net_checkpointing_non_meta_2.py \
# --pinn_outer_loss \
# --inner_opt_all_model_weights \
# --use_adam_inner_opt \
# --emb_type pde_const_emb \
# --relative_data_loss \
# --outer_loss_choice mse \
# --inner_crit_compare_to pde_zero \
# --image_width 128 \
# --g_dim 128 \
# --z_dim 64 \
# --dataset 1d_advection_multiparam \
# --data_root $DATADIR \
# --num_trials 1 \
# --n_past 2 \
# --n_future 2 \
# --num_threads 0 \
# --inner_crit_mode mse \
# --enc_dec_type vgg \
# --num_epochs_per_val 1 \
# --fno_modes 16 \
# --fno_width 64 \
# --fno_layers 3 \
# --emb_dim 64 \
# --val_batch_size 1 \
# --train_num_inner_steps 1 \
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
# --log_dir /data/divyam123/results_noether_summer/advection_multiparam/data_true_outer_learned_inner/PINNS_no_norm/ \
# --warmstart_emb_path best_ckpt_advection_multiparam_mse.pt \
# --channels 1 \
# --random_weights \
# --batch_norm_to_group_norm \
# --model_path ./checkpoints/pdes/t_past2/batch_5d \

SEED=1
LOSS=mse
python train_noether_net_checkpointing_non_meta_2_final_metrics.py \
--seed $SEED \
--frozen_val_emb \
--frozen_train_emb \
--tailor \
--use_cn \
--use_adam_inner_opt \
--emb_type pde_emb \
--relative_data_loss \
--outer_loss_choice mse \
--inner_crit_compare_to pde_zero \
--image_width 128 \
--g_dim 128 \
--z_dim 64 \
--dataset 1d_advection_multiparam \
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
--log_dir /data/divyam123/results_noether_summer/1d_advection_new_params/slurm_baseline_4k/seed=$SEED/no_norm_steps=5/run_data_outer_train=frozen_val=frozen/ \
--reload_dir /data/divyam123/slurm_runs/results_noether_summer/1d_advection_new_params/ckpt/seed=$SEED/no_norm_steps=5/run_data_outer_train=frozen_val=frozen/final/ckpt_model.pt \
--channels 1 \
--random_weights \
--batch_norm_to_group_norm \
--model_path ./checkpoints/pdes/t_past2/batch_5d \

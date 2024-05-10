CUDA_VISIBLE_DEVICES=6
DIR=/home/divyam123/noether_work_cpy/noether-networks/noether-networks/config_files/1d_burgers_multiparam_new/baseline_4k/new_residual_experiments_15

bash $DIR/run_data_outer_train=learnable_val=learnable_param_loss_init_frames.sh
bash $DIR/run_data.sh
bash $DIR/run_pinns.sh
bash $DIR/run_pino.sh
bash $DIR/run_true_conditioning.sh
bash $DIR/run_true_conditioning_single_scale_shift.sh

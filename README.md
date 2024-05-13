# Convex Optimization (EE227B) Final Project
This repository contains the code needed to reproduce the Noether network results in our literature review. The repository was forked from the original Noether network paper authors, and the vast majority of the code has remained the same. The main change we implemented was the dataloader for the 2D reaction diffusion equation, found in data/twod_reacdiff.py. The visualizations of the learned embeddings can be produced by running plot_embeddings.ipynb. 



# Noether Networks for video prediction

This directory contains code to train and evaluate a Noether Network for video prediction on the
Physics 101 dataset. Much of the model and utility code comes directly from the
[SVG codebase](https://github.com/edenton/svg) (Denton and Fergus); we use SVG as our baseline
video prediction model.

First, download the Physics 101 dataset, available at the Physics 101 [project
page](http://phys101.csail.mit.edu/), and unzip it in the `./data/phys101/` directory. The ramp
scenario data should be located at `./data/phys101/phys101/scenarios/ramp/`. You can do this by
running
```
./download_extract_phys101.sh
```

Ensure you have all the required dependencies, you can install them with the following command:
```
pip install requirements.txt
```

Then, train a Noether Network with the training script. For example, to train from scratch with a
single inner step, you can run the following command:
```
python train_noether_net.py \
    --image_width 128 \
    --g_dim 128 \
    --z_dim 64 \
    --dataset phys101 \
    --data_root ./data/phys101/phys101/scenarios/ramp \
    --tailor \
    --num_trials 1 \
    --n_past 2 \
    --n_future 20 \
    --num_threads 6 \
    --ckpt_every 2 \
    --inner_crit_mode mse \
    --enc_dec_type vgg \
    --emb_type conserved \
    --num_epochs_per_val 1 \
    --emb_dim 64 \
    --batch_size 2 \
    --num_inner_steps 1 \
    --num_jump_steps 0 \
    --n_epochs 1000 \
    --train_set_length 311 \
    --test_set_length 78 \
    --inner_lr .0001 \
    --val_inner_lr .0001 \
    --outer_lr 0.0001 \
    --outer_opt_model_weights \
    --random_weights \
    --only_twenty_degree \
    --frame_step 2 \
    --center_crop 1080 \
    --num_emb_frames 2 \
    --horiz_flip \
    --batch_norm_to_group_norm \
    --reuse_lstm_eps \
    --log_dir ./results/phys101/<experiment_id>/
```
where `<experiment_id>` specifies the subdirectory where the model checkpoints and tensorboard logs
will be written.

To train a baseline model, pass in `--num_inner_steps 0`.

To evaluate, run the evaluation script, passing in the model checkpoint you want to use:
```
python evaluate_noether_net.py \
    --model_path ./results/phys101/<experiment_id>/model_400.pth \
    --num_inner_steps 1 \
    --n_future 20 \
    --horiz_flip \
    --test_set_length 78 \
    --train_set_length 311 \
    --val_inner_lr .0001 \
    --reuse_lstm_eps \
    --data_root ./data/phys101/phys101/scenarios/ramp \
    --dataset phys101 \
    --n_past 2 \
    --tailor \
    --n_trials 1 \
    --only_twenty_degree \
    --frame_step 2 \
    --crop_upper_right 1080 \
    --center_crop 1080 \
    --batch_size 2 \
    --image_width 128 \
    --num_threads 4
```
You can pass `--adam_inner_opt` to use Adam instead of SGD in the inner loop.
This script will run the evaluation script, compute metrics on the test set, and cache these
metrics as numpy arrays.

You can load and plot the metrics with the `generate_figures.ipynb` notebook, which also contains
code to generate Grad-CAM heatmaps.


DG: UPDATE 05-10-2024


STEP 1: training non-pino methods:

config_files -> system of choice -> ckpt -> pick bash file and run
    a) for noether,pinns, data, true_conditioning run train_noether_net.py
    b) for PINO, run train_noether_net_checkpointing_pino_manual.py

STEP 2: computing final values + plots

config_files -> system of choice -> run_inference folders -> pick bash file and run
    a) for noether,pinns, data, true_conditioning run train_noether_net_final_inference.py
    b) for PINO, run train_noether_net_checkpointing_pino_final_inference.py

Each final inference run will save a 'saved_dictionary.pkl' which contains the mean 
and the batch variance of the inference run


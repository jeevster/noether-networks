#!/bin/bash

### Only difference to run.sh is the arguments you can use
HELP="script usage: $(basename $0) [-b batch size (default 10)] [-t Tailoring flag] [-c Cuda device (default '')] n_past n_future"
DATADIR=/data/nithinc/PDEs/2D/diffusion-reaction
BATCHSIZE=10
TAILORING=''
TAILORINGHUMAN=''
LOGDIR='./results/2d_reacdiff_pdeemb_FNO'
CUDA_VISIBLE_DEVICES=''

while getopts hb:tc: OPTION
do
    case "$OPTION" in
        h)
            echo "$HELP" >&2
            exit 1
            ;;
        b)
            BATCHSIZE="$OPTARG"
            ;;
        t)
            TAILORING='--tailor'
            TAILORINGHUMAN='_tailor'
            ;;
        c)
            CUDA_VISIBLE_DEVICES="$OPTARG"
            ;;
        ?)
            echo "$HELP" >&2
            exit 1
            ;;
    esac
done

NPAST="${@:$OPTIND:1}"
NFUTURE="${@:$OPTIND+1:1}"

[ -z "$NPAST" ] && echo $HELP >&2 && exit 1
[ -z "$NFUTURE" ] && echo $HELP >&2 && exit 1

echo "Args:"
echo "Number past frames: $NPAST"
echo "Number future frames: $NFUTURE"
echo "Batch size: $BATCHSIZE"
echo "GPU: $CUDA_VISIBLE_DEVICES"
if [ -z $TAILORING ]
then
    echo "Tailoring: False"
else
    echo "Tailoring: True"
fi

#changes relative to vanilla Noether: --inner_opt_all_model_weights, no prior/posterior so no kl loss
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_noether_net.py \
--image_width 128 \
--g_dim 128 \
--z_dim 64 \
--dataset 2d_reacdiff \
--data_root $DATADIR \
--num_trials 1 \
--n_past $NPAST \
--n_future $NFUTURE \
--num_threads 0 \
--ckpt_every 10 \
--inner_crit_mode mse \
--enc_dec_type vgg \
--emb_type conserved \
--num_epochs_per_val 1 \
--fno_modes 16 \
--fno_width 128 \
--fno_layers 2 \
--emb_dim 64 \
--pde_emb \
--batch_size $BATCHSIZE \
--num_inner_steps 1 \
--num_jump_steps 0 \
--n_epochs 100 \
--train_set_length 1000 \
--test_set_length 200 \
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
--log_dir "${LOGDIR}/past${NPAST}_future${NFUTURE}_train1000_val200_lr0.0001_bs${BATCHSIZE}${TAILOR}/" \
--channels 2 \
$TAILORING \
--random_weights \
--inner_opt_all_model_weights \
--batch_norm_to_group_norm \
--model_path ./checkpoints/pdes/t_past2/batch_5d

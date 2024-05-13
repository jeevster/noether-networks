DIR=/home/divyam123/noether_work_cpy/noether-networks/noether-networks/config_files/1d_burgers_multiparam_new/run_inference_4k_set
for FILE in $DIR/new_residual_experiments/*
do
    bash $FILE
done;
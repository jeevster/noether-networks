BASEDIR=config_files/1d_advection_multiparam_new/ckpt
for FILE in $BASEDIR/pinns_sweep_no_slurm/*; do
    echo ${FILE}
    bash ${FILE}
    sleep 1 # pause to be kind to the scheduler
done

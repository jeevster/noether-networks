GPU=9
FILE_PATH=config_files/1d_advection_multiparam_new/baseline_4k/pinns_sweep
CUDA_VISIBLE_DEVICES=${GPU} bash $FILE_PATH/run_data.sh
CUDA_VISIBLE_DEVICES=${GPU} bash $FILE_PATH/run_pinns_datareg=0.01_pdereg=1.sh
CUDA_VISIBLE_DEVICES=${GPU} bash $FILE_PATH/run_pinns_datareg=1_pdereg=0.01.sh
CUDA_VISIBLE_DEVICES=${GPU} bash $FILE_PATH/run_pinns_datareg=1_pdereg=1.sh
CUDA_VISIBLE_DEVICES=${GPU} bash $FILE_PATH/run_pinns_datareg=1_pdereg=10.sh
CUDA_VISIBLE_DEVICES=${GPU} bash $FILE_PATH/run_pinns_datareg=10_pdereg=1.sh



GPU=5
FILE_PATH=config_files/1d_diffusion_reaction_multiparam_new/baseline_4k/pinns_sweep_2
CUDA_VISIBLE_DEVICES=${GPU} bash $FILE_PATH/run_data.sh
CUDA_VISIBLE_DEVICES=${GPU} bash $FILE_PATH/run_pinns_datareg=1_pdereg=0.5.sh
CUDA_VISIBLE_DEVICES=${GPU} bash $FILE_PATH/run_pinns_datareg=1_pdereg=0.3.sh
CUDA_VISIBLE_DEVICES=${GPU} bash $FILE_PATH/run_pinns_datareg=1_pdereg=0.01.sh
CUDA_VISIBLE_DEVICES=${GPU} bash $FILE_PATH/run_pinns_datareg=1_pdereg=0.1.sh
CUDA_VISIBLE_DEVICES=${GPU} bash $FILE_PATH/run_pinns_datareg=1_pdereg=1.sh
import subprocess
import os
from glob import glob



# PATH = 'config_files/advection_multiparam_new/baseline'
GPU = 7
# PATH = 'config_files/1d_advection_multiparam_new/baseline_2'#'config_files/1d_advection_multiparam_new/ckpt'
# result = [y for x in oss.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
# for file_path in results:
#     if 'pino' in file_path:# or 'residual' in file_path:
#         print("FILEPATH", file_path)
#         subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)s

for seed in [0,1,2]:
    # PATH = f'config_files/1d_diffusion_reaction_multiparam_new/baseline_4k/seed={seed}_param/run_data_outer_train=learnable_val=learnable_param_loss_init_frames_no_pretraining.sh'
    # subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)
    PATH = f'config_files/1d_diffusion_reaction_multiparam_new/baseline_4k/seed={seed}_param/run_data_outer_train=learnable_val=learnable_init_frames_no_pretraining.sh'
    subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)

# /home/divyam123/noether_work_cpy/noether-networks/noether-networks/config_files/1d_diffusion_reaction_multiparam_new/baseline_4k/seed=2_param/run_data_outer_train=learnable_val=learnable_init_frames_no_pretraining.sh
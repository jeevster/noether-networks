import subprocess
import os
from glob import glob



# PATH = 'config_files/advection_multiparam_new/baseline'
# PATH = 'config_files/1d_diffusion_reaction_multiparam_new/baseline_4k/new_residual_experiments'#'config_files/1d_advection_multiparam_new/ckpt'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
# for file_path in result:
#     if 'learnable' in file_path:# or 'residual' in file_path:
#         print("FILEPATH", file_path)
#         subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)

# PATH = 'config_files/1d_burgers_multiparam_new/baseline_4k/new_residual_experiments'#'config_files/1d_advection_multiparam_new/ckpt'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
# for file_path in result:
#     if 'learnable' in file_path:# or 'residual' in file_path:
#         print("FILEPATH", file_path)
#         subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)

GPU = 6

PATH = 'config_files/1d_advection_multiparam_new/baseline_4k/new_residual_experiments'#'config_files/1d_advection_multiparam_new/ckpt'
result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
for file_path in result:
    print("FILEPATH", file_path)
    subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)

# for seed in [0,1,2]:
#     # PATH = f'config_files/1d_diffusion_reaction_multiparam_new/baseline_4k/seed={seed}_param/run_data_outer_train=learnable_val=learnable_param_loss_init_frames_no_pretraining.sh'
#     # subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)
#     PATH = f'config_files/1d_diffusion_reaction_multiparam_new/baseline_4k/seed={seed}_param/run_data_outer_train=learnable_val=learnable_init_frames_no_pretraining.sh'
#     subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)
# PATH = 'config_files/1d_burgers_multiparam_new/ckpt/new_residual_experiments/run_data_outer_train=learned_val=learned_param_loss_init_frames.sh'
# subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)
# PATH = 'config_files/1d_advection_multiparam_new/ckpt/data_restricted/run_pinns.sh'
# subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)
# PATH = 'config_files/1d_diffusion_reaction_multiparam_new/ckpt/new_residual_experiments/run_data_outer_train=learned_val=learned_param_loss_init_frames.sh'
# subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)

# /home/divyam123/noether_work_cpy/noether-networks/noether-networks/config_files/1d_diffusion_reaction_multiparam_new/baseline_4k/seed=2_param/run_data_outer_train=learnable_val=learnable_init_frames_no_pretraining.sh
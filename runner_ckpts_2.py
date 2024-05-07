import subprocess
import os
from glob import glob



# PATH = 'config_files/advection_multiparam_new/baseline'
# PATH = 'config_files/1d_advection_multiparam_new/ckpt'
# PATH = 'config_files/1d_advection_multiparam_new/baseline_2'#'config_files/1d_advection_multiparam_new/ckpt'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
# for file_path in result:
#     if 'run_true_conditioning' in file_path:# or 'residual' in file_path:
#         print("FILEPATH", file_path)
#         subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)

# PATH = 'config_files/1d_burgers_multiparam_new/baseline_2/seed=0'#'config_files/1d_advection_multiparam_new/ckpt'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
# for file_path in result:
#     if 'run_true_conditioning' in file_path:# or 'residual' in file_path:
#         print("FILEPATH", file_path)
#         subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)

GPU = 5
# # PATH = 'config_files/1d_advection_multiparam_new/ckpt'
# PATH = 'config_files/1d_diffusion_reaction_multiparam_new/baseline_4k/seed=0_param'#'config_files/1d_advection_multiparam_new/ckpt'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
# for file_path in result:
#     subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)
# PATH = 'config_files/1d_diffusion_reaction_multiparam_new/baseline_4k/seed=1_param'#'config_files/1d_advection_multiparam_new/ckpt'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
# for file_path in result:
#     subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)
#     PATH = 'config_files/1d_diffusion_reaction_multiparam_new/baseline_4k/seed=2_param'#'config_files/1d_advection_multiparam_new/ckpt'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
# for file_path in result:
#     subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)
# for seed in [0,1,2]:
#     PATH = f'config_files/1d_advection_multiparam_new/ckpt/param_seed={seed}/run_data.sh'
#     subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)
# for seed in [0,1,2]:
# PATH = f'config_files/1d_diffusion_reaction_multiparam_new/ckpt/new_residual_experiments'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
# print(result)
# for file_path in result:
#     if 'pino' not in file_path:
#         subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)
PATH = 'config_files/1d_diffusion_reaction_multiparam_new/ckpt/new_residual_experiments/run_data_outer_train=learnable_val=learnable_param_loss_init_frames.sh'
subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)

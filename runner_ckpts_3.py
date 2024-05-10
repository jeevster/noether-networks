import subprocess
import os
from glob import glob



# PATH = 'config_files/advection_multiparam_new/baseline'
GPU = 6
# for seed in [0,1,2]:
#     PATH = f'config_files/1d_burgers_multiparam_new/baseline_4k/seed={seed}_param'
#     result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
#     for file_path in result:
#         subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)
# for seed in [0,1,2]:
#     PATH = f'config_files/1d_burgers_multiparam_new/baseline_4k/embedding/run_advection_seed={seed}_param_loss.sh'
#     result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
#     for file_path in result:
#         subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)

# subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash config_files/1d_diffusion_reaction_multiparam_new/ckpt/param_seed=0/run_data.sh',shell=True)

# for seed in [0,1,2]:
#     PATH = f'config_files/1d_diffusion_reaction_multiparam_new/ckpt/param_seed={seed}/run_data_outer_train=learned_val=learned_param_loss_init_frames_no_pretraining.sh'
#     subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)

#     PATH = f'config_files/1d_diffusion_reaction_multiparam_new/ckpt/param_seed={seed}/run_data_outer_train=learned_val=learned_init_frames_no_pretraining.sh'
    # PATH = f'config_files/1d_diffusion_reaction_multiparam_new/ckpt/param_seed={seed}/run_data_outer_train=learned_val=learned_param_loss_init_frames_no_pretraining.sh'
    # subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)
# PATH = f'config_files/1d_advection_multiparam_new/ckpt/new_residual_experiments'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
# print(result)
# for file_path in result:
#     if 'pino' not in file_path:
#         subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)

# PATH = f'config_files/1d_burgers_multiparam_new/ckpt/new_residual_experiments'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
# print(result)
# for file_path in result:
#     if 'pino' not in file_path:
#         subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)

# for seed in [0,1,2]:
#     PATH = f'config_files/1d_advection_multiparam_new/ckpt/param_seed={seed}/run_data_outer_train=learned_val=learned_param_loss_init_frames_no_pretraining.sh'
#     subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)

#     PATH = f'config_files/1d_advection_multiparam_new/ckpt/param_seed={seed}/run_data_outer_train=learned_val=learned_init_frames_no_pretraining.sh'
#     subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)

# for seed in [0,1,2]:
#     PATH = f'config_files/1d_burgers_multiparam_new/ckpt/param_seed={seed}/run_data_outer_train=learned_val=learned_param_loss_init_frames_no_pretraining.sh'
#     subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)

#     PATH = f'config_files/1d_burgers_multiparam_new/ckpt/param_seed={seed}/run_data_outer_train=learned_val=learned_init_frames_no_pretraining.sh'
#     subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {PATH}',shell=True)
GPU = 2
PATH = 'config_files/1d_advection_multiparam_new/ckpt/pinns_sweep_no_slurm'#'config_files/1d_advection_multiparam_new/ckpt'
result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
for file_path in result:
    if 'seed=0' in file_path:# or 'residual' in file_path:
        print("FILEPATH", file_path)
        subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)

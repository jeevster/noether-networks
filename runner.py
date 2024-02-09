import subprocess
import os
from glob import glob

# # PATH = 'config_files/2d_react_diff'
# PATH = 'config_files/1d_diffusion_reaction_multiparam_new'
# # PATH = 'config_files/burgers_multiparam_new'

# # PATH = 'config_files/advection_multiparam_new'
# GPU = 7
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]

# def check(path):
#     # 2,3,4,5,
#     for rf in ['run_1', 'run_2','run_3','run_4','run_5']:
#         if rf in path:
#             return True
#     return False

# for file_path in result:
#     if 'noether_no_norm_step=5' in file_path and check(file_path): 
#         subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)


# PATH = 'config_files/2d_react_diff'
# PATH = 'config_files/1d_diffusion_reaction_multiparam_new'
# PATH = 'config_files/burgers_multiparam_new'

PATH = 'config_files/advection_multiparam_new'
GPU = 3
result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]

def check(path):
    # 2,3,4,5,
    for rf in ['run_5','run_6','run_7']:
        if rf in path:
            return True
    return False

for file_path in result:
    if 'no_norm' in file_path and check(file_path): 
        subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)

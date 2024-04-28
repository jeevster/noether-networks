import subprocess
import os
from glob import glob



# PATH = 'config_files/advection_multiparam_new/baseline'
GPU = 6

PATH = 'config_files/1d_advection_multiparam_new/baseline_2'
result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]

for file_path in result:
    if 'frozen' in file_path:
        subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)


PATH = 'config_files/1d_burgers_multiparam_new/baseline_2_cleaned'
result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]

for file_path in result:
    if 'frozen' in file_path:
        subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)


PATH = 'config_files/1d_diffusion_reaction_multiparam_new/baseline_2'
result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]

for file_path in result:
    if 'frozen' in file_path:
        subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)



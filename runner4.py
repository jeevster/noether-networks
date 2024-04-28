# import subprocess
# import os
# from glob import glob



# # PATH = 'config_files/advection_multiparam_new/baseline'
# GPU = 2

# PATH = 'config_files/2d_react_diff_copy/baseline_2'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]

# for file_path in result:
#     subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)



# #########
    
# PATH = 'config_files/2d_react_diff_copy/baseline_2_fixed'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]

# for file_path in result:
#     subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)

# #######
# PATH = 'config_files/2d_react_diff_copy/baseline_5'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]

# for file_path in result:
#     subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)

# ######
# PATH = 'config_files/2d_react_diff_copy/baseline_1024'
# result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]

# for file_path in result:
#     subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)

    

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import linalg

files = os.listdir('/data/divyam123/burgers_1024_2048_-2_0.3')
collection = list()
repeated = list()
files_list = set()
for file_name in files:
    data = h5py.File(f'/data/divyam123/burgers_1024_2048_-2_0.3/{file_name}','r')
    data_tensor = data['tensor']
    unwanted_indices = []
    for i in range(100):
        norm = np.linalg.norm(data_tensor[i,-1])
        if norm < 1e-5:
            # files_list.add(file_name)
            # collection.append((file_name, i))
            unwanted_indices.append(i)
        if np.sum(data_tensor[i,-1] == data_tensor[i,-2]) > 0:
            # repeated.append((file_name, i))
            unwanted_indices.append(i)

    os.makedirs('/data/divyam123/burgers_1024_2048_-2_0.3_clean', exist_ok = True)
    data_tensor = np.delete(data_tensor, unwanted_indices, axis = 0)
    if data_tensor.shape[0] > 0:
        print(data_tensor.shape, file_name)
        f = h5py.File(f'/data/divyam123/burgers_1024_2048_-2_0.3_clean/{file_name}', 'w')
        f['x-coordinate'] = data['x-coordinate'][:]
        f['t-coordinate'] = data['t-coordinate'][:]
        f['tensor'] = data_tensor
        f.close()

collection
import subprocess
import os
from glob import glob



GPU = 5
PATH = 'config_files/1d_burgers_multiparam_new/ckpt'
result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.sh'))]
for file_path in result:
    if 'embedding' in file_path and 'param_loss' in file_path:# or 'residual' in file_path:
        subprocess.run(f'CUDA_VISIBLE_DEVICES={GPU} bash {file_path}',shell=True)

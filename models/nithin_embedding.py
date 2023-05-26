import numpy as np
import h5py as h5
import time
import torch

def reaction_diff_2d_residual(output_path, residual_path, sim_config):
    dataset = h5.File(output_path, mode='r')
    residual_dataset = None
    while residual_dataset is None:
        try:
            residual_dataset = h.File(residual_path, mode='a')
        except IOError:
            residual_dataset = None
            time.sleep(.1)

    
    for seed in dataset.keys():
        data = np.asarray(dataset[seed]['data'])
        x = np.asarray(dataset[seed]['grid/x'])
        y = np.asarray(dataset[seed]['grid/y'])
        t = np.asarray(dataset[seed]['grid/t'])
        residuals = reaction_diff_2d_residual_compute(
            data[:, :, :, 0], data[:, :, :, 1], x, y, t, sim_config.k, sim_config.Du, sim_config.Dv)
        key = f'k={sim_config.k}_Du={sim_config.Du}_Dv={sim_config.Dv}/{seed}'
        residual_dataset.create_dataset(
            key, data=residuals, dtype='float32', compression='lzf')

    dataset.close()
    residual_dataset.close()

def partials_torch(data, x, y, t):
    y_axis = -1
    x_axis = -2
    t_axis = -3
    data_x = torch.gradient(data, spacing = (x,), dim=x_axis)[0]
    data_xx = torch.gradient(data_x, spacing = (x,), dim=x_axis)[0]
    data_y = torch.gradient(data, spacing = (y,), dim=y_axis)[0]
    data_yy = torch.gradient(data_y, spacing = (y,), dim=y_axis)[0]
    data_t = torch.gradient(data, spacing = t, dim=t_axis)[0]
    return data_x, data_y, data_xx, data_yy, data_t

def partials(data, x, y, t):
    y_axis = 2
    x_axis = 1
    t_axis = 0
    data_x = np.gradient(data, x, axis=x_axis)
    data_xx = np.gradient(data_x, x, axis=x_axis)
    data_y = np.gradient(data, y, axis=y_axis)
    data_yy = np.gradient(data_y, y, axis=y_axis)
    data_t = np.gradient(data, t, axis=t_axis)
    return data_x, data_y, data_xx, data_yy, data_t


def reaction_diff_2d_residual_compute(u, v, x, y, t, k, du, dv, return_partials = False):
    k = k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    du = du.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    dv = dv.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    u_x, u_y, u_xx, u_yy, u_t = partials_torch(u, x, y, t)
    v_x, v_y, v_xx, v_yy, v_t = partials_torch(v, x, y, t)
    # else:
    #     u_x, u_y, u_xx, u_yy, u_t = partials(u, x, y, t)
    #     v_x, v_y, v_xx, v_yy, v_t = partials(v, x, y, t)

    #2d reaction diffusion equations
    ru = u - (u ** 3) - k - v
    rv = u - v
    eqn1 = du * u_xx + du * u_yy + ru - u_t
    eqn2 = dv * v_xx + dv * v_yy + rv - v_t

    pde_residual = (eqn1 + eqn2).abs().mean(dim = (1,2,3))
    if return_partials:
        u_partials = torch.cat([u_x, u_y, u_xx, u_yy, u_t], dim = 1)
        v_partials = torch.cat([v_x, v_y, v_xx, v_yy, v_t], dim = 1)
        return pde_residual, torch.stack([u_partials, v_partials], dim = 2) #keep u and v partials separate
    else:
        return (eqn1 + eqn2).abs().mean(dim = (1,2,3))
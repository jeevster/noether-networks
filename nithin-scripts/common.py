import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import os

class ReactionDiff2D:

    @classmethod
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

    @classmethod
    def residual(u, v, x, y, t, k, du, dv):
        u_x, u_y, u_xx, u_yy, u_t = ReactionDiff2D.partials(u, x, y, t)
        v_x, v_y, v_xx, v_yy, v_t = ReactionDiff2D.partials(v, x, y, t)
        ru = u - (u ** 3) - k - v
        rv = u - v
        eqn1 = du * u_xx + du * u_yy + ru - u_t
        eqn2 = dv * v_xx + dv * v_yy + rv - v_t
        return np.abs(eqn1) + np.abs(eqn2)
    
    @jax.jit
    @classmethod
    def residual_jax(u, v, u_partials, v_partials, k, du, dv):
            u_x, u_y, u_xx, u_yy, u_t = u_partials
            v_x, v_y, v_xx, v_yy, v_t = v_partials
            ru = u - (u ** 3) - k - v
            rv = u - v
            eqn1 = du * u_xx + du * u_yy + ru - u_t
            eqn2 = dv * v_xx + dv * v_yy + rv - v_t
            res = jnp.abs(eqn1) + jnp.abs(eqn2)
            return jnp.mean(res, axis=(1, 2))

    @classmethod
    def sensitivity_analysis(u, v, x, y, t, gt_k, gt_du, gt_dv, min_noise=-.5, max_noise=1.5, num_noise=20):
        u_partials = ReactionDiff2D.partials(u, x, y, t)
        v_partials = ReactionDiff2D.partials(v, x, y, t)
        noise = jnp.linspace(min_noise, max_noise, num_noise)
        gt_residual = ReactionDiff2D.residual_jax(
            u, v, u_partials, v_partials, gt_k, gt_du, gt_dv)
        gt_k = jnp.asarray(gt_k)
        gt_du = jnp.asarray(gt_du)
        gt_dv = jnp.asarray(gt_dv)
        k = jnp.broadcast_to(gt_k, (noise.shape[0])) + noise
        du = jnp.broadcast_to(gt_du, (noise.shape[0])) + noise
        dv = jnp.broadcast_to(gt_dv, (noise.shape[0])) + noise
        k_residuals = jax.vmap(ReactionDiff2D.residual_jax,
                            in_axes=(None, None, None, None, 0, None, None))(u, v, u_partials, v_partials, k, gt_du, gt_dv)
        du_residuals = jax.vmap(ReactionDiff2D.residual_jax,
                                in_axes=(None, None, None, None, None, 0, None))(u, v, u_partials, v_partials, gt_k, du, gt_dv)
        dv_residuals = jax.vmap(ReactionDiff2D.residual_jax,
                                in_axes=(None, None, None, None, None, None, 0))(u, v, u_partials, v_partials, gt_k, gt_du, dv)
        return gt_residual, du_residuals, dv_residuals, k_residuals, noise

    @classmethod
    def plot_residuals(residuals, t, x, k, du, dv, savedir):
        #residuals: n_seeds x n_t x n_x x n_y
        #k: float
        #du: float
        #dv: float
        residuals = np.mean(residuals, axis=(2, 3))
        for seed in range(residuals.shape[0]):
            plt.plot(t, residuals[seed, :])
        plt.yscale('log')
        plt.xlabel('Time')
        plt.ylabel('Residual')
        plt.title('k = ' + str(k) + ', du = ' + str(du) + ', dv = ' + str(dv) + f', tdim = {t.shape[0]}, resolution = {x.shape[0]}')
        plt.savefig(savedir + '/k=' + str(k) + '_du=' + str(du) + '_dv=' + str(dv) + f'_tdim={t.shape[0]}_resolution={x.shape[0]}.png')

    @classmethod
    def plot_sensitivity_analysis(gt_residuals, du_residuals, dv_residuals, k_residuals, noise, k, du, dv, savdir):
        # gt_residuals: n_timesteps, n_spatial
        # x_residuals: n_noise, n_timesteps, n_spatial
        # noise: n_noise

        # First plot du noise
        ReactionDiff2D.plot_sensitivity_inner(gt_residuals, du_residuals, noise, 'du', savdir, k, du, dv)
        ReactionDiff2D.plot_sensitivity_inner(gt_residuals, dv_residuals, noise, 'dv', savdir, k, du, dv)
        ReactionDiff2D.plot_sensitivity_inner(gt_residuals, k_residuals, noise, 'k', savdir, k, du, dv)
    
    @classmethod
    def plot_sensitivity_inner(gt_residuals, noise_residuals, noise, param, save_dir, k, du, dv):
        plt.plot(gt_residuals, label='Ground Truth', linestyle='-.')
        for i in range(len(noise_residuals)):
            plt.plot(noise_residuals[i], label=f'{noise[i]:.2e}')
        plt.yscale('log')
        plt.xlabel('Time')
        plt.ylabel('Residual')
        plt.title(f'k = {k}, du = {du}, dv = {dv} - {param} noise')
        # plt.legend()
        plt.legend(loc="upper right", borderaxespad=0)
        plt.savefig(os.path.join(save_dir, f'{param}_noise.png'))
        plt.clf()


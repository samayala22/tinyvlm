import sys

if sys.platform == "win32":
    sys.path.append(r".\\build\\windows\\x64\\release")
elif sys.platform == "linux":
    sys.path.append(r"./build/linux/x86_64/release")
else:
    exit()

import numpy as np
import scipy as sp
from dataclasses import dataclass

# local imports
import dof2
import helpers
import continuation as cont
import harmonic_balance as hb
import plotting as plot

from libhbvlm import HBVLM

INITIAL_ONLY = False

np.set_printoptions(
    linewidth=200, # max line width
    formatter={'float': '{:.3e}'.format} # format shortE
) 

@dataclass
class System:
    M      : callable
    C      : callable
    K      : callable
    dMdU   : callable
    dCdU   : callable
    dKdU   : callable
    fnlt   : callable        # time‐domain NL force
    fnlf   : callable        # frequency‐domain NL force

def create_motion_system() -> System:
    def fnlt(t, X, u, v, omega, U):
        return np.array([
            - (ndv.omega / U)**2 * u[0],
            - 1/(U**2) * torsional_func(u[1])
        ])

    def fnlf(X, omega, U):
        forces_t = np.zeros_like(X)
        hbvlm.run(omega, X, forces_t)
        forces_t[0, :] = - forces_t[0, :] / (np.pi * ndv.mu)
        forces_t[1, :] = (2.0 * forces_t[1, :]) / (np.pi * ndv.mu * ndv.r_a**2)
        return forces_t
    
    def M(U):
        return np.array([
            [1.0, ndv.x_a],
            [ndv.x_a / ndv.r_a**2, 1.0]
        ])
    
    def C(U):
        return np.array([
            [2.0 * ndv.zeta_h * ndv.omega / U, 0.0],
            [0.0, 2.0 * ndv.zeta_a / U]
        ])
    
    def K(U):
        return np.zeros((2,2))
    
    def dMdU(U):
        return np.zeros((2, 2))
    
    def dCdU(U):
        return - (1.0 / U**2) * np.array([
            [2.0 * ndv.zeta_h * ndv.omega, 0.0],
            [0.0, 2.0 * ndv.zeta_a]
        ])
    
    def dKdU(U):
        return np.zeros((2, 2))
    
    return System(M, C, K, dMdU, dCdU, dKdU, fnlt, fnlf)

if __name__ == "__main__":
    torsional_spring = 1
    torsional_spring_names = ["Freeplay", "Cubic", "Linear"]

    if (torsional_spring == 0):
        torsional_func = dof2.alpha_freeplay
    elif (torsional_spring == 1):
        torsional_func = dof2.alpha_cubic
    else:
        torsional_func = dof2.alpha_linear

    # Independent params
    dims = hb.Dims(
        n_d=2,          # number of degrees of freedom
        n_h=10          # number of harmonics
    )

    vars_b = 0.5 # half chord
    flutter_speed = 6.285
    flutter_ratio_start = 0.3
    flutter_ratio_end = 0.8

    hbvlm = HBVLM("cpu", "./mesh/infinite_rectangular_10x1.x")
    hbvlm.init(dims.n_h, 1.0/vars_b)

    # Dependent params
    # param_start = flutter_speed * flutter_ratio_start
    # param_end = flutter_speed * flutter_ratio_end
    param_start = 10.0
    param_end = 20.0
    # Time integration
    t_final = 2000.0
    dt = 0.2
    ndv = dof2.NDVars(
        a_h = -0.5,
        omega = 0.2,
        zeta_a = 0.0,
        zeta_h = 0.0,
        x_a = 0.25,
        mu = 100.0,
        r_a = 0.5,
        U = param_start
    )

    y0 = np.array([0, np.radians(3), 0, 0, 0, 0]) # h, a, hd, ad, x1, x2
    system, _ = dof2.create_monolithic_system(y0, ndv, torsional_func)
    sol = sp.integrate.solve_ivp(system, (0, t_final), y0, t_eval=np.arange(0, t_final, dt), method='RK45')
    
    idx_start = int(0.75 * len(sol.t))
    t_tr = sol.t[idx_start:]
    u_tr = sol.y[0:2, idx_start:]   # shape = (n_dofs, N_tr)

    u_coeffs, omega0 = hb.truncated_series_approximation(dt, u_tr, dims)
    
    X0 = np.zeros(dims.n_d * dims.n_c + 2)
    X0[:-2] = u_coeffs.T.reshape(-1)
    X0[-2] = omega0
    X0[-1] = param_start

    # X0[2] = 5e-3
    # X0[3] = 5e-3
    # X0[-2] = 0.085
    # X0[-1] = param_start

    # z_ref = np.zeros_like(X0)
    # z_ref[-1] = 1.0
    # J_fd = extended_residual_jacobian(X0, X0, z_ref, hb_residual, Parametrisation.Local)
    # J_hy = extended_residual_jacobian_hybrid(X0, X0, z_ref, Parametrisation.Local)
    
    # print("FD cond: ", np.linalg.cond(J_fd))
    # print("Hybrid cond: ", np.linalg.cond(J_hy))

    # np.testing.assert_allclose(J_fd, J_hy)
 
    metadata = cont.Metadata()
    metadata.name = f"2DOF {torsional_spring_names[torsional_spring]}"
    metadata.param_start = param_start
    metadata.param_end = param_end
    metadata.max_steps = 1 if INITIAL_ONLY else 10000
    metadata.scaling = True
    metadata.step_adapt = True
    metadata.ds = 0.01
    metadata.dims = dims
    
    metadata = cont.continuation(X0, create_motion_system, metadata)
    
    if helpers.getenv("PLOT"):
        X_mat = metadata.X
        if X_mat.shape[1] == 1:
            hb_sol_t, hb_sol0 = hb.to_timedomain(0.0, t_final, dt, dims.n_d, X_mat[:-2, 0], X_mat[-2, 0], dims.n_h)
            fig = plot.create_dofs_figure(["Heave", "Pitch"])
            # dof2.plot_uvlm(fig)
            dof2.plot_monolithic(fig, sol)
            plot.add_data_and_psd(fig, hb_sol_t, hb_sol0[0, :], "HB-VLM", 1, 1, 3)
            plot.add_data_and_psd(fig, hb_sol_t, hb_sol0[1, :], "HB-VLM", 3, 1, 3)
            plot.add_data_and_psd(fig, hb_sol_t, hb_sol0[2, :], "HB-VLM", 1, 2, 3)
            plot.add_data_and_psd(fig, hb_sol_t, hb_sol0[3, :], "HB-VLM", 3, 2, 3)
            
            plot.fig_save(fig, f"build/2dof/hbvlm0")
        else:
            cont.plot_hb_continuation(metadata)

# Notes:
# - Dimitriadis introduction to nonlinear aeroelasticity p333: jacobian sign changes during continuation
# - Reproduce fig 7.14 ?

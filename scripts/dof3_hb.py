import sys

if sys.platform == "win32":
    sys.path.append(r".\\build\\windows\\x64\\debug")
elif sys.platform == "linux":
    sys.path.append(r"./build/linux/x86_64/release")
else:
    exit()

import numpy as np
import scipy as sp
from dataclasses import dataclass

# local imports
import dof3
import helpers
import continuation as cont
import harmonic_balance as hb
import plotting as plot

from libhbvlm3 import HBVLM

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
    def fnlt(u, u_dot, omega, U):
        return np.array([
            0.0, # h
            0.0, # alpha
            - ((v.omega_beta / v.omega_alpha)**2 * v.r_beta**2) * torsional_func(u[2]) # beta
        ])

    def fnlf(X, omega, U):
        assert omega > 0.0, "Omega must be positive"
        V = U / (v.b * v.omega_alpha)
        forces_t = np.zeros_like(X)
        hbvlm.run(omega, U, X, forces_t)
        forces_t[0, :] = - V*V * forces_t[0, :] / (np.pi*v.mu)
        forces_t[1, :] = 2.0*V*V * forces_t[1, :] / (np.pi*v.mu)
        forces_t[2, :] = (1-v.c)**2 * V * V * forces_t[2, :] / (2.0 * np.pi * v.mu)
        return forces_t
    
    def M(U):
        M_s = np.zeros((3,3))
        M_s[0, 0] = v.m_t / v.m
        M_s[0, 1] = v.x_alpha
        M_s[0, 2] = v.x_beta
        M_s[1, 0] = v.x_alpha
        M_s[1, 1] = v.r_alpha**2
        M_s[1, 2] = (v.c - v.a) * v.x_beta + v.r_beta**2
        M_s[2, 0] = v.x_beta
        M_s[2, 1] = (v.c - v.a) * v.x_beta + v.r_beta**2
        M_s[2, 2] = v.r_beta**2
        return M_s
    
    def C(U):
        D_s = np.zeros((3,3))
        D_s[0, 0] = v.sigma * v.zeta_h
        D_s[1, 1] = v.r_alpha**2 * v.zeta_alpha
        D_s[2, 2] = (v.omega_beta / v.omega_alpha) * v.r_beta**2 * v.zeta_beta
        D_s = 2 * D_s
        return D_s
    
    def K(U):
        K_s = np.zeros((3,3))
        K_s[0, 0] = v.sigma**2
        K_s[1, 1] = v.r_alpha**2
        # K_s[2, 2] = (v.omega_beta / v.omega_alpha)**2 * v.r_beta**2
        return K_s
    
    def dMdU(U): return np.zeros((3, 3))
    def dCdU(U): return np.zeros((3, 3))
    def dKdU(U): return np.zeros((3, 3))
    
    return System(M, C, K, dMdU, dCdU, dKdU, fnlt, fnlf)

if __name__ == "__main__":
    torsional_spring = 0
    torsional_spring_names = ["Freeplay", "Cubic", "Linear"]

    if (torsional_spring == 0):
        torsional_func = dof3.alpha_freeplay
    # elif (torsional_spring == 1):
    #     torsional_func = dof3.alpha_cubic
    else:
        torsional_func = dof3.alpha_linear

    v = dof3.Vars()
    v.a = -0.5 
    v.b = 0.127 
    v.c = 0.5 
    v.I_alpha = 0.01347
    v.I_beta = 0.0003264
    v.k_h = 2818.8
    v.k_alpha = 37.34
    v.k_beta = 3.9
    v.m = 1.5666
    v.m_t = 3.39298
    v.r_alpha = 0.7321
    v.r_beta = 0.1140
    v.S_alpha = 0.08587
    v.S_beta = 0.00395
    v.x_alpha = 0.4340
    v.x_beta = 0.02
    v.omega_h = 42.5352
    v.omega_alpha = 52.6506
    v.omega_beta = 109.3093
    v.rho = 1.225
    v.zeta_h = 0.0113
    v.zeta_alpha = 0.01626
    v.zeta_beta = 0.0115
    v.sigma = v.omega_h / v.omega_alpha
    v.V = v.U / (v.b * v.omega_alpha)
    v.mu = v.m / (np.pi * v.rho * v.b**2)

    # Independent params
    dims = hb.Dims(
        n_d=3,          # number of degrees of freedom
        n_h=5          # number of harmonics
    )

    hbvlm = HBVLM("cpu", ["mesh/3dof_wing_9x5.x", "mesh/3dof_flap_3x5.x"])
    hbvlm.init(dims.n_h, 1.0/v.b)

    # Params
    flutter_speed = 23.9
    # param_start = flutter_speed * 0.3
    # param_end = flutter_speed * 0.6
    param_start = 7.0
    param_end = 10.0

    # Time integration
    t_final = 2000.0
    dt = 0.2
    vec_t = np.arange(0, t_final, dt)
    y0 = np.zeros(8, dtype=np.float64) # hd, ad, bd, h, a, b, x1, x2
    y0[3] = 0.01 / v.b # h
    system = dof3.AeroelasticSystem(v, True)
    sol = sp.integrate.solve_ivp(system.coupled_system, (0, t_final), y0, t_eval=vec_t, method='RK45')

    idx_start = int(0.75 * len(sol.t))
    t_tr = sol.t[idx_start:]
    u_tr = sol.y[3:6, idx_start:]   # shape = (n_dofs, N_tr)
    u_coeffs, omega0 = hb.truncated_series_approximation(dt, u_tr, dims)
    X0 = np.zeros(dims.n_d * dims.n_c + 2)
    X0[:-2] = u_coeffs.T.reshape(-1)
    X0[-2] = omega0
    X0[-1] = param_start

    # X0[2] = 5e-3
    # X0[3] = 5e-3
    # X0[-2] = 0.085
    # X0[-1] = param_start
 
    metadata = cont.Metadata()
    metadata.name = f"3DOF {torsional_spring_names[torsional_spring]}"
    metadata.param_start = param_start
    metadata.param_end = param_end
    metadata.max_steps = 1
    metadata.scaling = False
    metadata.step_adapt = True
    metadata.ds = [0.05]
    metadata.dims = dims
    
    metadata = cont.continuation(X0, create_motion_system, metadata)
    
    if helpers.getenv("PLOT"):
        X_mat = metadata.X
        if X_mat.shape[1] == 1:
            hb_sol_t, hb_sol0 = hb.to_timedomain(0.0, t_final, dt, dims.n_d, X_mat[:-2, 0], X_mat[-2, 0], dims.n_h)
            aero_forces = system.aero_forces(sol.y)
            fig = plot.create_dofs_figure(["Heave", "Pitch", "Control"])
            dof3.plot_solution(fig, aero_forces, sol)

            plot.add_data_and_psd(fig, hb_sol_t, hb_sol0[0, :], "HB-VLM", 1, 1, 3)
            plot.add_data_and_psd(fig, hb_sol_t, np.degrees(hb_sol0[1, :]), "HB-VLM", 3, 1, 3)
            plot.add_data_and_psd(fig, hb_sol_t, np.degrees(hb_sol0[2, :]), "HB-VLM", 5, 1, 3)

            plot.add_data_and_psd(fig, hb_sol_t, hb_sol0[3, :], "HB-VLM", 1, 2, 3)
            plot.add_data_and_psd(fig, hb_sol_t, np.degrees(hb_sol0[4, :]), "HB-VLM", 3, 2, 3)
            plot.add_data_and_psd(fig, hb_sol_t, np.degrees(hb_sol0[5, :]), "HB-VLM", 5, 2, 3)
            
            plot.fig_save(fig, f"build/3dof/hbvlm0")
        else:
            cont.plot_hb_continuation(metadata)

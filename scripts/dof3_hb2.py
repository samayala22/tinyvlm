import numpy as np
import scipy as sp
from dataclasses import dataclass

# local imports
import dof3
import helpers
import continuation as cont
import harmonic_balance as hb
import finite_diff as fd
import plotting as plot

BETA_NL_DAMPING = False
INITIAL_ONLY = True

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
 
def nonlinear_damping(A, omega):
    """
    A: Amplitude
    omega: frequency in Hz
    """
    a3 = -1.1
    a2 = 2.0
    olog = np.log10(omega)
    return v.zeta_beta * np.max(a3 * olog**3 + a2 * olog**2 + 1, 0.2)

def create_motion_system() -> System:
    def fnlt(t, X, u, u_dot, omega, U):
        f = np.zeros((8, t.shape[0]))
        gamma = 0
        f[0, :] = v.mu * u[3, :] * v.sigma**2 * (1 + gamma * u[3, :]**2)
        f[2, :] = v.mu * ((v.omega_beta / v.omega_alpha)**2 * v.r_beta**2) * dof3.alpha_freeplay(u[5, :])      
        return -f

    def fnlf(X, omega, U):
        return np.zeros_like(X)
    
    def M(U):
        M_s = np.zeros((8,8))
        return M_s
    
    def C(U):
        v.U = U
        v.V = v.U / (v.b * v.omega_alpha)
        sys = dof3.AeroelasticSystem(v, True)
        M2 = np.zeros((8, 8))

        M2[0:3, 0:3] = sys.M_s - sys.M_a
        M2[3:6, 3:6] = np.eye(3)
        M2[6:8, 6:8] = np.eye(2)
        M2[6:8, 0:3] = - sys.Q_a
        return M2
    
    def K(U):
        v.U = U
        v.V = v.U / (v.b * v.omega_alpha)
        sys = dof3.AeroelasticSystem(v, True)

        M1 = np.zeros((8, 8))
        M1[0:3, 0:3] = sys.D_a - sys.D_s
        M1[0:3, 3:6] = sys.K_a - sys.K_s
        M1[0:3, 6:8] = sys.L_delta
        M1[3:6, 0:3] = np.eye(3)
        M1[6:8, 0:3] = sys.Q_v
        M1[6:8, 6:8] = sys.L_lambda
        return -M1
    
    def dMdU(U): return np.zeros((8, 8))
    def dCdU(U): return fd.cd2(C, U)
    def dKdU(U): return fd.cd2(K, U)
    
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

    # Params
    flutter_speed = 23.9
    # param_start = flutter_speed * 0.3
    # param_end = flutter_speed * 0.6
    param_start = 6.0
    param_end = 22.0

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
    v.U = param_start
    v.V = v.U / (v.b * v.omega_alpha)
    v.mu = v.m / (np.pi * v.rho * v.b**2)

    # Independent params
    dims = hb.Dims(
        n_d=8,          # number of degrees of freedom
        n_h=10          # number of harmonics
    ) 

    # Time integration
    t_final = 4000.0
    dt = 0.2 
    vec_t = np.arange(0, t_final, dt)
    y0 = np.zeros(8, dtype=np.float64) # hd, ad, bd, h, a, b, x1, x2
    y0[3] = 0.01 / v.b # h
    system = dof3.AeroelasticSystem(v, True)
    sol = sp.integrate.solve_ivp(system.coupled_system, (0, t_final), y0, t_eval=vec_t, method='RK45')

    idx_start = int(0.9 * len(sol.t))
    t_tr = sol.t[idx_start:]
    u_tr = sol.y[:, idx_start:]   # shape = (n_dofs, N_tr)
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
    metadata.max_steps = 1 if INITIAL_ONLY else 10000
    metadata.scaling = False
    metadata.step_adapt = True
    metadata.ds = 0.02
    metadata.dims = dims
    
    motion = create_motion_system()
    metadata = cont.continuation(X0, motion, metadata)
    
    if helpers.getenv("PLOT"):
        X_mat = metadata.X
        if X_mat.shape[1] == 1:
            hb_sol_t, hb_sol0 = hb.to_timedomain(vec_t, dims.n_d, X_mat[:-2, 0], X_mat[-2, 0], dims.n_h)
            aero_forces = system.aero_forces(sol.y)
            fig = plot.create_dofs_figure(["Heave", "Pitch", "Control"])
            dof3.plot_solution(fig, aero_forces, sol, v)

            plot.add_data_and_psd(fig, hb_sol_t, hb_sol0[3, :], "HB-VLM", 1, 1, 3)
            plot.add_data_and_psd(fig, hb_sol_t, np.degrees(hb_sol0[4, :]), "HB-VLM", 3, 1, 3)
            plot.add_data_and_psd(fig, hb_sol_t, np.degrees(hb_sol0[5, :]), "HB-VLM", 5, 1, 3)

            dof3.format_plot(fig)
            plot.fig_save(fig, f"build/3dof/hbvlm0")
        else:
            cont.plot_hb_continuation(metadata)

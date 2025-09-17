import sys
import numpy as np
import scipy as sp
from dataclasses import dataclass

# local imports
import helpers
import continuation as cont
import harmonic_balance as hb
import plotting as plot
import newmark

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
    def fnlt(t, X, u, u_dot, omega, U):
        k_nl0 = 1.0
        k_nl1 = 1.0
        F0 = 2

        return np.array([
            F0 * np.cos(omega * t) - k_nl0 * u[0]**3, # + 0.5 * u[0]**2
            - k_nl1 * u[1]**3
        ])

    def fnlf(X, omega, U):
        forces_t = np.zeros_like(X)
        return forces_t
    
    def M(omega):
        m1 = 1.0
        m2 = 1.0
        M_s = np.array([
            [m1, 0.0],
            [0.0, m2]
        ])
        return M_s
    
    def C(omega):
        zeta1 = 0.1
        zeta2 = 0.1
        C_s = np.array([
            [zeta1, 0.0],
            [0.0, zeta2]
        ])
        return C_s
    
    def K(omega):
        k1 = 1
        k2 = 1
        k12 = 5
        K_s = np.array([
            [k1 + k12, -k12],
            [-k12, k2 + k12]
        ])
        return K_s
    
    def dMdU(U): return np.zeros((2, 2))
    def dCdU(U): return np.zeros((2, 2))
    def dKdU(U): return np.zeros((2, 2))
    
    return System(M, C, K, dMdU, dCdU, dKdU, fnlt, fnlf)

if __name__ == "__main__":
    INITIAL_ONLY = 0
    omega_beg = 0.05
    omega_end = 5.0
    
    # Independent params
    dims = hb.Dims(
        n_d=2,          # number of degrees of freedom
        n_h=15          # number of harmonics
    )

    # Time integration
    motion = create_motion_system()
    t_final = 2000.0
    dt = 0.2
    vec_t = np.arange(0, t_final, dt)
    u0 = np.zeros(2, dtype=np.float64)
    u0[0] = 0.1
    def nl_func(t, u, v): return motion.fnlt(t, None, u, v, omega_beg, None)
    t, u, v, a = newmark.nonlinear_newmark_solve(
        motion.M(None),
        motion.C(None),
        motion.K(None),
        u0,
        np.zeros_like(u0),
        nl_func,
        t_final,
        dt
    )

    idx_start = int(0.75 * len(t))
    t_tr = t[idx_start:]
    u_tr = u[:, idx_start:]
    u_coeffs, omega0 = hb.truncated_series_approximation(dt, u_tr, dims)
    X0 = np.zeros(dims.n_d * dims.n_c + 1)
    X0[:-1] = u_coeffs.T.reshape(-1)
    X0[-1] = omega_beg

    metadata = cont.Metadata()
    metadata.name = f"2DOF Duffing Oscillators"
    metadata.param_start = omega_beg
    metadata.param_end = omega_end
    metadata.max_steps = 1 if INITIAL_ONLY else 10000
    metadata.scaling = True
    metadata.step_adapt = True
    metadata.ds = 0.02
    metadata.dims = dims
    
    metadata = cont.continuation(X0, create_motion_system, metadata)
    
    if not helpers.getenv("PLOT"):
        sys.exit(0)
    
    X_mat = metadata.X
    if X_mat.shape[1] == 1:
        hb_sol_t, hb_sol0 = hb.to_timedomain(vec_t, dims.n_d, X_mat[:-1, 0], X_mat[-1, 0], dims.n_h)
        _, hb_sol_approx = hb.to_timedomain(vec_t, dims.n_d, X0[:-1], omega0, dims.n_h)

        fig = plot.create_dofs_figure(["dof1", "dof2"])
        plot.add_data_and_psd(fig, t, u[0, :], "Time Integration", 1, 1)
        plot.add_data_and_psd(fig, t, u[1, :], "Time Integration", 3, 1)
        plot.add_data_and_psd(fig, t, v[0, :], "Time Integration", 1, 2)
        plot.add_data_and_psd(fig, t, v[1, :], "Time Integration", 3, 2)

        plot.add_data_and_psd(fig, t, hb_sol_approx[0, :], "Approx", 1, 1, 1)
        plot.add_data_and_psd(fig, t, hb_sol_approx[1, :], "Approx", 3, 1, 1)
        plot.add_data_and_psd(fig, t, hb_sol_approx[2, :], "Approx", 1, 2, 1)
        plot.add_data_and_psd(fig, t, hb_sol_approx[3, :], "Approx", 3, 2, 1)

        plot.add_data_and_psd(fig, t, hb_sol0[0, :], "HB", 1, 1, 2)
        plot.add_data_and_psd(fig, t, hb_sol0[1, :], "HB", 3, 1, 2)
        plot.add_data_and_psd(fig, t, hb_sol0[2, :], "HB", 1, 2, 2)
        plot.add_data_and_psd(fig, t, hb_sol0[3, :], "HB", 3, 2, 2)
        plot.fig_save(fig, f"build/duffing/test")
        
    else:
        cont.plot_hb_continuation(metadata)

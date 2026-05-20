import numpy as np
import scipy as sp
from dataclasses import dataclass
import sys

# local imports
import dof3
import helpers
import continuation as cont
import harmonic_balance as hb
import finite_diff as fd
import plotting as plot

BETA_NL_DAMPING = False
INITIAL_ONLY = False
 
def nonlinear_damping(A, omega):
    """
    A: Amplitude
    omega: frequency in Hz
    """
    a3 = -1.1
    a2 = 2.0
    olog = np.log10(omega)
    return v.zeta_beta * np.max(a3 * olog**3 + a2 * olog**2 + 1, 0.2)

class AeroelasticSystem:
    def __init__(self, v):
        self.M_s = np.zeros((3,3)) # dimensionless structural intertia matrix
        self.D_s = np.zeros((3,3)) # dimensionless structural damping matrix
        self.K_s = np.zeros((3,3)) # dimensionless structural stiffness matrix

        self.v = v

        self.M_s[0, 0] = v.m_t / v.m
        self.M_s[0, 1] = v.x_alpha
        self.M_s[0, 2] = v.x_beta
        self.M_s[1, 0] = v.x_alpha
        self.M_s[1, 1] = v.r_alpha**2
        self.M_s[1, 2] = (v.c - v.a) * v.x_beta + v.r_beta**2
        self.M_s[2, 0] = v.x_beta
        self.M_s[2, 1] = (v.c - v.a) * v.x_beta + v.r_beta**2
        self.M_s[2, 2] = v.r_beta**2
        self.M_s = v.mu * self.M_s

        self.D_s[0, 0] = v.sigma * v.zeta_h
        self.D_s[1, 1] = v.r_alpha**2 * v.zeta_alpha
        self.D_s[2, 2] = (v.omega_beta / v.omega_alpha) * v.r_beta**2 * v.zeta_beta
        self.D_s = 2 * v.mu * self.D_s

        # K_s[0, 0] = sigma**2 * (1 + gamma * y[3]**2)
        self.K_s[1, 1] = v.r_alpha**2
        # K_s[2, 2] = (v.omega_beta / v.omega_alpha)**2 * v.r_beta**2
        self.K_s = v.mu * self.K_s

        sqrt_1_minus_c2 = np.sqrt(1 - v.c**2)
        acos_c = np.arccos(v.c)

        T1 = (-1/3) * sqrt_1_minus_c2 * (2 + v.c**2) + v.c * acos_c
        T2 = v.c * (1 - v.c**2) - sqrt_1_minus_c2 * (1 + v.c**2) * acos_c + v.c * (acos_c)**2
        T3 = -((1/8) + v.c**2) * (acos_c)**2 + (1/4) * v.c * sqrt_1_minus_c2 * acos_c * (7 + 2*v.c**2) - (1/8) * (1 - v.c**2) * (5 * v.c**2 + 4)
        T4 = -acos_c + v.c * sqrt_1_minus_c2
        T5 = -(1 - v.c**2) - (acos_c)**2 + 2 * v.c * sqrt_1_minus_c2 * acos_c
        T6 = T2
        T7 = -((1/8) + v.c**2) * acos_c + (1/8) * v.c * sqrt_1_minus_c2 * (7 + 2 * v.c**2)
        T8 = (-1/3) * sqrt_1_minus_c2 * (2 * v.c**2 + 1) + v.c * acos_c
        T9 = (1/2) * ((1/3) * (sqrt_1_minus_c2)**3 + v.a * T4)
        T10 = sqrt_1_minus_c2 + acos_c
        T11 = acos_c * (1 - 2 * v.c) + sqrt_1_minus_c2 * (2 - v.c)
        T12 = sqrt_1_minus_c2 * (2 + v.c) - acos_c * (2 * v.c + 1)
        T13 = (1/2) * (-T7 - (v.c - v.a) * T1)
        T14 = (1/16) + (1/2) * v.a * v.c

        self.M_a = np.zeros((3,3)) # dimensionless aerodynamic inertia matrix
        self.D_a = np.zeros((3,3)) # dimensionless aerodynamic damping matrix
        self.K_a = np.zeros((3,3)) # dimensionless aerodynamic stiffness matrix
        self.L_delta = np.zeros((3, 2)) # dimensionless aero lagging matrix
        self.Q_a = np.zeros((2, 3)) # matrix of terms multiplied by the modal accelerations
        self.Q_v = np.zeros((2, 3)) # matrix of terms multiplied by the modal velocities
        self.L_lambda = np.zeros((2, 2))

        self.M_a[0, 0] = -1
        self.M_a[0, 1] = v.a
        self.M_a[0, 2] = T1 / np.pi
        self.M_a[1, 0] = v.a
        self.M_a[1, 1] = -(1/8 + v.a**2)
        self.M_a[1, 2] = -2 * T13 / np.pi
        self.M_a[2, 0] = T1 / np.pi
        self.M_a[2, 1] = -2 * T13 / np.pi
        self.M_a[2, 2] = T3 / (np.pi**2)

        self.D_a[0, 0] = (-2)
        self.D_a[0, 1] = (-2 * (1 - v.a))
        self.D_a[0, 2] = (T4 - T11) / np.pi
        self.D_a[1, 0] = (1 + 2 * v.a)
        self.D_a[1, 1] = (v.a * (1 - 2 * v.a))
        self.D_a[1, 2] = (T8 - T1 + (v.c - v.a) * T4 + v.a * T11) / np.pi
        self.D_a[2, 0] = (-T12 / np.pi)
        self.D_a[2, 1] = (2 * T9 + T1 + (T12 - T4) * (v.a - 0.5)) / np.pi
        self.D_a[2, 2] = (T11 * (T4 - T12)) / (2 * np.pi**2)

        self.K_a[0, 0] = 0
        self.K_a[0, 1] = (-2)
        self.K_a[0, 2] = (-2 * T10) / np.pi
        self.K_a[1, 0] = 0
        self.K_a[1, 1] = (1 + 2 * v.a)
        self.K_a[1, 2] = (2 * v.a * T10 - T4) / np.pi
        self.K_a[2, 0] = 0
        self.K_a[2, 1] = (-T12) / np.pi
        self.K_a[2, 2] = (-1 / np.pi**2) * (T5 - T10 * (T4 - T12))
        
        # Difference lies between coeffs of R.T Jones and W.P Jones
        # Yung p220
        d1 = 0.165
        d2 = 0.335
        # l1 = 0.041
        # l2 = 0.320
        l1 = 0.0455
        l2 = 0.300

        self.L_delta[0, 0] = d1
        self.L_delta[0, 1] = d2
        self.L_delta[1, 0] = - (0.5 + v.a) * d1
        self.L_delta[1, 1] = - (0.5 + v.a) * d2
        self.L_delta[2, 0] = T12 * d1 / (2 * np.pi)
        self.L_delta[2, 1] = T12 * d2 / (2 * np.pi)

        self.Q_a[0, 0] = 1
        self.Q_a[0, 1] = 0.5 - v.a
        self.Q_a[0, 2] = T11 / (2 * np.pi)
        self.Q_a[1, 0] = 1
        self.Q_a[1, 1] = 0.5 - v.a
        self.Q_a[1, 2] = T11 / (2 * np.pi)

        self.Q_v[0, 1] = 1
        self.Q_v[0, 2] = T10 / np.pi
        self.Q_v[1, 1] = 1
        self.Q_v[1, 2] = T10 / np.pi

        self.L_lambda[0, 0] = - l1
        self.L_lambda[1, 1] = - l2

        self.M2 = np.zeros((8, 8))
        self.M2[0:3, 0:3] = self.M_s - self.M_a
        self.M2[3:6, 3:6] = np.eye(3)
        self.M2[6:8, 6:8] = np.eye(2)
        self.M2[6:8, 0:3] = - self.Q_a

def create_motion_system(v, system, torsional_func):
    def fnlt(t, X, u, u_dot, omega, U):
        f = np.zeros((8, t.shape[0]))
        gamma = 0
        f[0, :] = v.mu * u[3, :] * v.sigma**2 * (1 + gamma * u[3, :]**2)
        f[2, :] = v.mu * ((v.omega_beta / v.omega_alpha)**2 * v.r_beta**2) * torsional_func(u[5, :])
        return -f

    def fnlf(X, omega, U):
        return np.zeros_like(X)
    
    def M(U):
        M_s = np.zeros((8,8))
        return M_s
    
    def C(U):
        return system.M2
    
    def K(U):
        V = U / (v.b * v.omega_alpha)
        D_a = V * system.D_a
        D_s = system.D_s
        K_a = V**2 * system.K_a
        K_s = system.K_s
        L_delta = 2 * V * system.L_delta
        Q_v = V * system.Q_v
        L_lambda = V * system.L_lambda

        M1 = np.zeros((8, 8))
        M1[0:3, 0:3] = D_a - D_s
        M1[0:3, 3:6] = K_a - K_s
        M1[0:3, 6:8] = L_delta
        M1[3:6, 0:3] = np.eye(3)
        M1[6:8, 0:3] = Q_v
        M1[6:8, 6:8] = L_lambda
        return -M1
    
    def dMdU(U): return np.zeros((8, 8))
    def dCdU(U): return fd.cd2(C, U)
    def dKdU(U): return fd.cd2(K, U)
    
    return cont.System(M, C, K, dMdU, dCdU, dKdU, fnlt, fnlf)

def run_continuation(torsional_spring:int, param_start:float, param_end:float, harmonics:int=5):
    assert torsional_spring in [0, 1, 2], "Invalid torsional spring type"
    torsional_name = ["freeplay", "cubic", "linear"][torsional_spring]
    torsional_func = [dof3.alpha_freeplay, dof3.alpha_poly, dof3.alpha_linear][torsional_spring]

    v = dof3.Vars()
    v = dof3.update_vars(v, param_start)
    dims = hb.Dims(
        n_d=8,          # number of degrees of freedom
        n_h=harmonics          # number of harmonics
    )

    # Time integration
    t_final = 1000.0
    dt = 0.1 
    vec_t = np.arange(0, t_final, dt)
    y0 = np.zeros(8, dtype=np.float64) # hd, ad, bd, h, a, b, x1, x2
    y0[3] = 0.01 / v.b # h
    system = dof3.AeroelasticSystem(v, True, torsional_func)
    sol = sp.integrate.solve_ivp(system.coupled_system, (0, t_final), y0, t_eval=vec_t, method='RK45')

    idx_start = int(0.9 * len(sol.t))
    t_tr = sol.t[idx_start:]
    u_tr = sol.y[:, idx_start:]   # shape = (n_dofs, N_tr)
    u_coeffs, omega0 = hb.truncated_series_approximation(dt, u_tr, dims)
    X0 = np.zeros(dims.n_d * dims.n_c + 2)
    X0[:-2] = u_coeffs.T.reshape(-1)
    X0[-2] = omega0
    X0[-1] = param_start

    metadata = cont.Metadata()
    metadata.name = f"3dof_{torsional_name}"
    metadata.param_start = param_start
    metadata.param_end = param_end
    metadata.max_steps = 1 if INITIAL_ONLY else 5000
    metadata.scaling = True
    metadata.step_adapt = True
    metadata.ds = 0.02
    metadata.dims = dims
    
    system2 = AeroelasticSystem(v)
    motion = create_motion_system(v, system2, torsional_func)
    metadata = cont.continuation(X0, motion, metadata)

    if helpers.getenv("PLOT"):
        X_mat = metadata.X
        hb_sol_t, hb_sol0 = hb.to_timedomain(vec_t, dims.n_d, X_mat[:-2, 0], X_mat[-2, 0], dims.n_h)
        aero_forces = system.aero_forces(sol.y)
        fig = plot.create_dofs_figure(["Heave", "Pitch", "Control"])
        dof3.plot_solution(fig, aero_forces, sol, v)

        plot.add_data_and_psd(fig, hb_sol_t, hb_sol0[3, :], "HB-VLM", 1, 1, 3)
        plot.add_data_and_psd(fig, hb_sol_t, np.degrees(hb_sol0[4, :]), "HB-VLM", 3, 1, 3)
        plot.add_data_and_psd(fig, hb_sol_t, np.degrees(hb_sol0[5, :]), "HB-VLM", 5, 1, 3)

        dof3.format_plot(fig)
        plot.fig_save(fig, f"build/3dof/hbvlm0", html=True, pdf=False)

if __name__ == "__main__":
    torsional_spring = 1
    argv = sys.argv
    assert len(argv) == 5, "Usage: python dof3_combined.py <torsional_spring_type> <param_start> <param_end> <harmonics>"

    torsional_spring = int(argv[1])
    param_start = float(argv[2])
    param_end = float(argv[3])
    harmonics = int(argv[4])
    run_continuation(torsional_spring, param_start, param_end, harmonics)

    # run_continuation(1, 8.0, 20.0, 5)

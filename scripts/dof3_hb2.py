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

def create_motion_system():
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
        return sys.M2
    
    def K(U):
        V = U / (v.b * v.omega_alpha)
        D_a = V * sys.D_a
        D_s = sys.D_s
        K_a = V**2 * sys.K_a
        K_s = sys.K_s
        L_delta = 2 * V * sys.L_delta
        Q_v = V * sys.Q_v
        L_lambda = V * sys.L_lambda

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

if __name__ == "__main__":
    torsional_spring = 1
    torsional_spring_names = ["freeplay", "cubic", "linear"]

    if (torsional_spring == 0):
        torsional_func = dof3.alpha_freeplay
    elif (torsional_spring == 1):
        torsional_func = dof3.alpha_poly
    else:
        torsional_func = dof3.alpha_linear

    # Params
    flutter_speed = 23.9
    # param_start = flutter_speed * 0.3
    # param_end = flutter_speed * 0.6
    param_start = 11.0
    param_end = 1.0

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

    sys = AeroelasticSystem(v)

    # Independent params
    dims = hb.Dims(
        n_d=8,          # number of degrees of freedom
        n_h=10          # number of harmonics
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

    # X0[2] = 5e-3
    # X0[3] = 5e-3
    # X0[-2] = 0.085
    # X0[-1] = param_start

    metadata = cont.Metadata()
    metadata.name = f"3dof_{torsional_spring_names[torsional_spring]}"
    metadata.param_start = param_start
    metadata.param_end = param_end
    metadata.max_steps = 1 if INITIAL_ONLY else 5000
    metadata.scaling = True
    metadata.step_adapt = True
    metadata.ds = 0.02
    metadata.dims = dims
    
    motion = create_motion_system()
    if not helpers.getenv("POST"):
        metadata = cont.continuation(X0, motion, metadata)
        exit(0)
    
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
            cont.plot_hb_continuation([metadata])

    rms_samples = 20
    rms_param = np.linspace(5.0, 20.0, rms_samples)
    rms_mat = np.zeros((dims.n_d, rms_samples))
    for i, U in enumerate(rms_param):
        v.U = U
        v.V = v.U / (v.b * v.omega_alpha)
        system = dof3.AeroelasticSystem(v, True, torsional_func)
        sol = sp.integrate.solve_ivp(system.coupled_system, (0, t_final), y0, t_eval=vec_t, method='RK45')

        idx_start = int(0.9 * len(sol.t))
        u_tr = sol.y[:, idx_start:]
        rms = np.sqrt(np.mean(u_tr**2, axis=1))
        rms_mat[:, i] = rms

    if torsional_spring == 1:
        metadata_files = [
            "build/cont_3dof_cubic_st_6_end_20_it_285.pkl",
            "build/cont_3dof_cubic_st_6_end_1_it_326.pkl",
            "build/cont_3dof_cubic_st_12_end_20_it_212.pkl",
            "build/cont_3dof_cubic_st_12_end_10_it_161.pkl",
            "build/cont_3dof_cubic_st_11_end_1_it_405.pkl" # went back and forth
        ]
    elif torsional_spring == 0:
        metadata_files = [
            "build/cont_3dof_freeplay_st_6_end_20_it_284.pkl",
            "build/cont_3dof_freeplay_st_6_end_1_it_808.pkl",
            "build/cont_3dof_freeplay_st_15_end_20_it_157.pkl",
            "build/cont_3dof_freeplay_st_15_end_1_it_495.pkl",
            "build/cont_3dof_freeplay_st_11_end_20_it_752.pkl",
            "build/cont_3dof_freeplay_st_11_end_9_it_85.pkl"
        ]
    
    metadatas = []
    import pickle
    for filename in metadata_files:
        with open(filename, 'rb') as f:
            metadatas.append(pickle.load(f))
        print(f"Loaded: {filename}")
    cont.plot_hb_continuation(metadatas, timeseries=(rms_param, rms_mat))

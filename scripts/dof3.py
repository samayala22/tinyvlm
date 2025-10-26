from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from tqdm import tqdm
import plotting as plot

@dataclass
class Vars:
    a: float = 0.0# Dimensionless distance between mid-chord and EA (-0.5)
    b: float = 0.0# Semi-chord (0.127 m)
    c: float = 0.0# Dimensionless distance between flap hinge and mid-chord (0.5)
    I_alpha: float = 0.0# Mass moment of inertia of the wing-flap about wing EA per unit span
    I_beta: float = 0.0# Mass moment of inertia of the flap about flap hinge line
    k_h: float = 0.0# linear structural stiffness coefficient of plunging
    k_alpha: float = 0.0# Linear structural stiffness coefficient of plunging
    k_beta: float = 0.0# Linear structural stiffness coefficient of pitching
    m: float = 0.0# Mass of wing-aileron per span
    m_t: float = 0.0# Mass of wing-aileron and the supports per span
    r_alpha: float = 0.0# dimensionless radius of gyration around elastic axis
    r_beta: float = 0.0# dimensionless radius of gyration around flap hinge axis
    S_alpha: float = 0.0# static mass moment of wing-flap about wing EA per unit span
    S_beta: float = 0.0# static mass moment of flap about flap hinge line per unit span
    x_alpha: float = 0.0# dimensionless distance between airfoil EA and the center of gravity
    x_beta: float = 0.0# dimensionless distance between flap center of gravity and flap hinge axis
    omega_h: float = 0.0# uncoupled plunge natural frequency
    omega_alpha: float = 0.0# uncoupled pitch natural frequency
    omega_beta: float = 0.0# uncoupled flap natural frequency
    rho: float = 0.0# fluid density
    zeta_h: float = 0.0# plunge damping ratio
    zeta_alpha: float = 0.0# pitch damping ratio
    zeta_beta: float = 0.0# flap damping ratio
    U: float = 0.0 # velocity
    sigma: float = 0.0
    V: float = 0.0
    mu: float = 0.0

def alpha_freeplay(alpha, M0=0.0, Mf=0.0, delta=np.radians(4.24), a_f=np.radians(-2.12)):
    """
    alpha: dof angle
    M0: preload
    Mf: freeplay region stiffness
    delta: freeplay region width
    a_f: freeplay region start angle
    """
    return np.where(
        alpha < a_f,
        M0 + alpha - a_f,
        np.where(
            alpha <= (a_f + delta),
            M0 + Mf * (alpha - a_f),
            M0 + alpha - a_f + delta * (Mf - 1)
        )
    )

def alpha_poly(alpha, c2 = 0.0, c3 = 1.0):
    return c2 * alpha**2 + c3 * alpha**3

def alpha_linear(alpha):
    return alpha

class AeroelasticSystem:
    def __init__(self, v: Vars, coupled=True):
        self.M_s = np.zeros((3,3)) # dimensionless structural intertia matrix
        self.D_s = np.zeros((3,3)) # dimensionless structural damping matrix
        self.K_s = np.zeros((3,3)) # dimensionless structural stiffness matrix

        self.v = v
        self.coupled = coupled

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
        self.D_a = v.V * self.D_a

        self.K_a[0, 0] = 0
        self.K_a[0, 1] = (-2)
        self.K_a[0, 2] = (-2 * T10) / np.pi
        self.K_a[1, 0] = 0
        self.K_a[1, 1] = (1 + 2 * v.a)
        self.K_a[1, 2] = (2 * v.a * T10 - T4) / np.pi
        self.K_a[2, 0] = 0
        self.K_a[2, 1] = (-T12) / np.pi
        self.K_a[2, 2] = (-1 / np.pi**2) * (T5 - T10 * (T4 - T12))
        self.K_a = v.V**2 * self.K_a
        
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
        self.L_delta = 2 * v.V * self.L_delta

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
        self.Q_v = v.V * self.Q_v

        self.L_lambda[0, 0] = - l1
        self.L_lambda[1, 1] = - l2
        self.L_lambda = v.V * self.L_lambda

    def yn_func(self, y_: np.ndarray):
        y = y_.reshape(y_.shape[0], -1) # so that it works for both y_ as vector or matrix
        yn = np.zeros_like(y)
        gamma = 0 # Cubic factor (0 for linear spring)
        yn[0, :] = self.v.mu * y[3, :] * self.v.sigma**2 * (1 + gamma * y[3, :]**2)
        yn[2, :] = self.v.mu * ((self.v.omega_beta / self.v.omega_alpha)**2 * self.v.r_beta**2) * alpha_freeplay(y[5, :])      
        return yn.reshape(y_.shape)

    def uncoupled_system(self, t, y: np.ndarray):
        M1 = np.zeros((8, 8))
        M2 = np.zeros((8, 8))

        M2[0:3, 0:3] = self.M_s
        M2[3:6, 3:6] = np.eye(3)
        M2[6:8, 6:8] = np.eye(2)
        M2[6:8, 0:3] = - self.Q_a

        M1[0:3, 0:3] = - self.D_s
        M1[0:3, 3:6] = - self.K_s
        M1[3:6, 0:3] = np.eye(3)
        M1[6:8, 0:3] = self.Q_v
        M1[6:8, 6:8] = self.L_lambda

        return np.linalg.solve(M2, M1 @ y - self.yn_func(y))

    def coupled_system(self, t, y: np.ndarray):
        # M2 @ dy/dt = M1 @ y + yn
        M1 = np.zeros((8, 8))
        M2 = np.zeros((8, 8))

        M2[0:3, 0:3] = self.M_s - self.M_a
        M2[3:6, 3:6] = np.eye(3)
        M2[6:8, 6:8] = np.eye(2)
        M2[6:8, 0:3] = - self.Q_a

        M1[0:3, 0:3] = self.D_a - self.D_s
        M1[0:3, 3:6] = self.K_a - self.K_s
        M1[0:3, 6:8] = self.L_delta
        M1[3:6, 0:3] = np.eye(3)
        M1[6:8, 0:3] = self.Q_v
        M1[6:8, 6:8] = self.L_lambda

        return np.linalg.solve(M2, M1 @ y - self.yn_func(y))

        # dy/dt = A @ y
        # A = np.zeros((8,8))
        # inv_inert_mat = np.linalg.inv(M_s - M_a)
        # A[0:3, 0:3] = - inv_inert_mat @ (D_s - D_a)
        # A[0:3, 3:6] = - inv_inert_mat @ (K_s - K_a)
        # A[0:3, 6:8] = inv_inert_mat @ L_delta
        # A[3:6, 0:3] = np.eye(3)
        # A[6:8, 0:3] = Q_a @ A[0:3, 0:3]+ Q_v
        # A[6:8, 3:6] = Q_a @ A[0:3, 3:6]
        # A[6:8, 6:8] = Q_a @ A[0:3, 6:8] + L_lambda

        # return A @ y

    def uncoupled_accel(self, y: np.ndarray):
        return np.linalg.solve(self.M_s, -self.D_s @ y[0:3, :] - self.K_s @ y[3:6, :] - self.yn_func(y)[3:6, :])
    
    def coupled_accel(self, y: np.ndarray):
        return np.linalg.solve(self.M_s-self.M_a, (self.D_a-self.D_s) @ y[0:3, :] + (self.K_a- self.K_s) @ y[3:6, :] - self.yn_func(y)[3:6, :] + self.L_delta @ y[6:8, :])

    def aero_forces(self, y: np.ndarray):
        accel = self.coupled_accel(y) if self.coupled else self.uncoupled_accel(y)
        forces_aero = self.M_a @ accel + self.D_a @ y[0:3, :] + self.K_a @ y[3:6, :] + self.L_delta @ y[6:8, :]
        # force_structure = self.M_s @ self.coupled_accel(y) + self.D_s @ y[0:3, :] + self.K_s @ y[3:6, :] + self.yn_func(y)[3:6, :]
        # assert np.isclose(force_structure, forces_aero).all()
        return forces_aero
    
def plot_solution(fig, aero_forces, mono, v):
    plot.add_data_and_psd(fig, mono.t, mono.y[3, :], "Theodorsen", 1, 1) # h
    plot.add_data_and_psd(fig, mono.t, mono.y[0, :], "Theodorsen", 1, 2) # dh
    plot.add_data_and_psd(fig, mono.t, aero_forces[0, :]/v.mu, "Theodorsen", 1, 3) # force h
    plot.add_data_and_psd(fig, mono.t, np.degrees(mono.y[4, :]), "Theodorsen", 3, 1) # alpha
    plot.add_data_and_psd(fig, mono.t, np.degrees(mono.y[1, :]), "Theodorsen", 3, 2) # dalpha
    plot.add_data_and_psd(fig, mono.t, aero_forces[1, :]/v.mu, "Theodorsen", 3, 3) # force alpha
    plot.add_data_and_psd(fig, mono.t, np.degrees(mono.y[5, :]), "Theodorsen", 5, 1) # beta
    plot.add_data_and_psd(fig, mono.t, np.degrees(mono.y[2, :]), "Theodorsen", 5, 2) # dbeta
    plot.add_data_and_psd(fig, mono.t, aero_forces[2, :]/v.mu, "Theodorsen", 5, 3) # force beta

def format_plot(fig):
    plot.format_subplot(fig, 1, 1, "", r"$\Large{h}$")
    plot.format_subplot(fig, 1, 2, "", r"$\Large{\dot{h}}$")
    plot.format_subplot(fig, 1, 3, "", "Lift")
    plot.format_subplot(fig, 2, 1, "", "")
    plot.format_subplot(fig, 2, 2, "", "")
    plot.format_subplot(fig, 2, 3, "", "")
    plot.format_subplot(fig, 3, 1, "", r"$\Large{\alpha}$")
    plot.format_subplot(fig, 3, 2, "", r"$\Large{\dot{\alpha}}$")
    plot.format_subplot(fig, 3, 3, "", "Moment")
    plot.format_subplot(fig, 4, 1, "", "")
    plot.format_subplot(fig, 4, 2, "", "")
    plot.format_subplot(fig, 4, 3, "", "")
    plot.format_subplot(fig, 5, 1, "", r"$\Large{\beta}$")
    plot.format_subplot(fig, 5, 2, "", r"$\Large{\dot{\beta}}$")
    plot.format_subplot(fig, 5, 3, "", "Moment")
    plot.format_subplot(fig, 6, 1, "", "")
    plot.format_subplot(fig, 6, 2, "", "")
    plot.format_subplot(fig, 6, 3, "", "")

if __name__ == "__main__":
    # Darabseh 2022
    # v = Vars(
    #     a = -0.5,
    #     b = 0.127, # m
    #     c = 0.5,
    #     I_alpha = 0.01347, # kgm
    #     I_beta = 0.0003264, # kgm
    #     k_h = 2818.8, # kg/ms^2
    #     k_alpha = 37.34, # kg/ms^2
    #     k_beta = 3.9, # kg/ms^2
    #     m = 1.558, # kg/m
    #     m_t = 3.3843, # kg/m
    #     r_alpha = 0.7321,
    #     r_beta = 0.1140,
    #     S_alpha = 0.08587, # kg
    #     S_beta = 0.00395, # kg
    #     x_alpha = 0.4340,
    #     x_beta = 0.02,
    #     omega_h = 42.5352, # Hz
    #     omega_alpha = 52.6506, # Hz
    #     omega_beta = 109.3093,
    #     rho = 1.225, # kg/m^3
    #     zeta_h = 0.0133,
    #     zeta_alpha = 0.01626,
    #     zeta_beta = 0.0133,
    #     U = 23.72 # m/s
    # )

    # U_vec = [0.49 * 23.9]
    # Interesting velocities:
    # U_vec = [12.7778] # LCO transition
    # U_vec = [12.63158]
    # U_vec = [12.36842] # Non symmetric 2 period LCO
    # U_vec = [12.75168] # Bifuracation reports many points ?
    U_vec = [7.0]
    # U_vec = np.linspace(0.2, 22.5, 100)

    peaks_data = [[], [], []]
    peaks_U = [[], [], []]
    coupled_sim = True

    for U in tqdm(U_vec): 
        # Conner
        v = Vars()
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
        v.U = U
        v.sigma = v.omega_h / v.omega_alpha
        v.V = v.U / (v.b * v.omega_alpha)
        v.mu = v.m / (np.pi * v.rho * v.b**2)
        U_str = f"{int(U)}"

        dt = 0.1
        # t_final = 5.0 * v.omega_alpha
        t_final = 250.0
        vec_t = np.arange(0, t_final, dt)
        n = len(vec_t)

        # hd, ad, bd, h, a, b, x1, x2
        y0 = np.zeros(8, dtype=np.float64)
        y0[3] = 0.01 / v.b # h

        system = AeroelasticSystem(v, coupled_sim)
        ivp = system.coupled_system if coupled_sim else system.uncoupled_system
        mono = solve_ivp(ivp, (0, t_final), y0, t_eval=vec_t, method='RK45')
        
        if (not mono.success):
            print("Monolithic solver failed")
            exit(1)

        if len(U_vec) == 1:
            aero_forces = system.aero_forces(mono.y)

            # Plotting
            fig = plot.create_dofs_figure(["Wing Heave", "Wing Pitch", "Flap Pitch"])
            plot_solution(fig, aero_forces, mono, v)

            uvlm_file = Path("build/windows/x64/release/3dof.txt")
            if uvlm_file.exists():
                with open(uvlm_file, "r") as f:
                    uvlm_tsteps = int(f.readline())
                    uvlm = np.zeros((10, uvlm_tsteps))
                    i = 0
                    for line in f:
                        uvlm[:, i] = np.array(list(map(float, line.split())))
                        i += 1
                    uvlm = uvlm[:, :i] # shrink in case the number of steps is not equal (stopped simulation)

                    plot.add_data_and_psd(fig, uvlm[0, :], uvlm[1, :], "UVLM", 1, 1, 1, dash="dot")
                    plot.add_data_and_psd(fig, uvlm[0, :], uvlm[4, :], "UVLM", 1, 2, 1, dash="dot")
                    plot.add_data_and_psd(fig, uvlm[0, :], uvlm[7, :], "UVLM", 1, 3, 1, dash="dot")
                    plot.add_data_and_psd(fig, uvlm[0, :], np.degrees(uvlm[2, :]), "UVLM", 3, 1, 1, dash="dot")
                    plot.add_data_and_psd(fig, uvlm[0, :], np.degrees(uvlm[5, :]), "UVLM", 3, 2, 1, dash="dot")
                    plot.add_data_and_psd(fig, uvlm[0, :], uvlm[8, :], "UVLM", 3, 3, 1, dash="dot")
                    plot.add_data_and_psd(fig, uvlm[0, :], np.degrees(uvlm[3, :]), "UVLM", 5, 1, 1, dash="dot")
                    plot.add_data_and_psd(fig, uvlm[0, :], np.degrees(uvlm[6, :]), "UVLM", 5, 2, 1, dash="dot")
                    plot.add_data_and_psd(fig, uvlm[0, :], uvlm[9, :], "UVLM", 5, 3, 1, dash="dot")

            format_plot(fig)
            plot.fig_save(fig, f"build/3dof/3dof_{U_str}", html=False, height=1500)

            # Poincare maps
            names = ["h", "alpha", "beta"]
            latex_names = [r"h", r"\alpha", r"\beta"]
            mono_idx_st = int(0.95 * mono.t.shape[0])
            uvlm_idx_st = int(0.95 * uvlm.shape[1])
            for i in range(3):
                fig = plot.fig_create_multi(1, 1)
                fig.add_trace(
                    go.Scatter(
                        x=mono.y[3+i, mono_idx_st:],
                        y=mono.y[i, mono_idx_st:],
                        mode="lines",
                        line = {"dash": "solid", "width": 5, "color": '#0066CC'},
                        name="Theodorsen"
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=uvlm[1+i, uvlm_idx_st:],
                        y=uvlm[4+i, uvlm_idx_st:],
                        mode="lines",
                        line = {"dash": "dot", "width": 5, "color": "#FF8400"},
                        name="UVLM"
                    )
                )
                plot.format_subplot(fig, 1, 1, r"$\Large{" + latex_names[i] + "}$", r"$\Large{\dot{" + latex_names[i] + "}}$")
                if i != 2: fig.update_layout(showlegend=False)
                plot.fig_save(fig, f"build/3dof/3dof_poincare_{U_str}_{names[i]}", html=False, height=500)

        else:
            slice_start = int(0.9 * n)
            peaks_slice = mono.y[3:6, slice_start:]
            
            for i in range(3):
                peaks_data[i].append(peaks_slice[i, plot.find_peak_idx(peaks_slice[i, :])])
                peaks_U[i].append(np.array([U] * peaks_data[i][-1].shape[0]))

    if len(U_vec) > 1:
        names = ["Heave", "Pitch", "Control"]
        fig = plot.fig_create_multi(3, 1)
        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate(peaks_U[i]), 
                    y=np.concatenate(peaks_data[i]), 
                    mode="markers",
                    marker = {"size": 3},
                    name = names[i]
                ),
                row=i + 1, 
                col=1
            )
            plot.format_subplot(fig, i + 1, 1, r"Reduced Velocity $V$", "Amplitude")
        
        fig.update_xaxes(showticklabels=False, title_text="", row=1, col=1)
        fig.update_xaxes(showticklabels=False, title_text="", row=2, col=1)

        plot.fig_save(fig, f"build/3dof/3dof_Cubic_bifurcation", html=False, height=1000)

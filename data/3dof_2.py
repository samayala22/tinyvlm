import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
import scipy as sp
from scipy.integrate import solve_ivp

@dataclass
class Vars:
    a: float  # Dimensionless distance between mid-chord and EA (-0.5)
    b: float  # Semi-chord (0.127 m)
    c: float  # Dimensionless distance between flap hinge and mid-chord (0.5); Plunge structural damping coefficient per unit span (1.7628 kg/ms)
    C_h: float  # Plunge structural damping coefficient per unit span (1.7628 kg/ms)
    C_alpha: float  # Pitch structural damping coefficient per unit span (0.0231 kgm/s)
    C_beta: float  # Flap structural damping coefficient per unit span (0.0008 kgm/s)
    I_alpha: float  # Mass moment of inertia of the wing-flap about wing EA per unit span (0.01347 kgm)
    I_beta: float  # Mass moment of inertia of the flap about the flap hinge line per unit span (0.0003264 kgm)
    k_h: float # linear strctural stiffness coefficient of plunging per unit span
    k_alpha: float  # Linear structural stiffness coefficient of plunging per unit span (2818.8 kg/ms²)
    k_beta: float  # Linear structural stiffness coefficient of pitching per unit span (37.34 kgm/s²); Linear structural stiffness coefficient of flap per unit span (3.9 kgm/s²); Mass of wing-aileron per span (1.558 kg/m)
    m: float  # Mass of wing-aileron per span
    m_t: float  # Mass of wing-aileron and the supports per span
    r_alpha: float # dimensionless radius of gyration around elastic axis
    r_beta: float # dimensionless radius of gyration around flap hinge axis
    S_alpha: float # static mass moment of wing-flap about wing EA per unit span
    S_beta: float # static mass moment of flap about flap hinge line per unit span
    x_alpha: float # dimensionless distance between airfoil EA and the center of gravity
    x_beta: float # dimensionless distance between flap center of gravity and flap hinge axis
    omega_h: float # uncoupled plunge natural frequency
    omega_alpha: float # uncoupled pitch natural frequency
    omega_beta: float # uncoupled flap natural frequency
    rho: float # fluid density
    zeta_h: float # plunge damping ratio
    zeta_alpha: float # pitch damping ratio
    zeta_beta: float # flap damping ratio
    U: float # velocity

def create_monolithic_system(y0: np.ndarray, v: Vars):
    def monolithic_system(t, y: np.ndarray):
        A = np.zeros((8,8))

        M_s = np.zeros((3,3)) # dimensionless structural intertia matrix
        D_s = np.zeros((3,3)) # dimensionless structural damping matrix
        K_s = np.zeros((3,3)) # dimensionless structural stiffness matrix

        sigma = v.omega_h / v.omega_alpha
        V = v.U / (v.b * v.omega_alpha)
        mu = v.m / (np.pi * v.rho * v.b**2)

        M_s[0, 0] = v.m_t / v.m
        M_s[0, 1] = v.x_alpha
        M_s[0, 2] = v.x_beta
        M_s[1, 0] = v.x_alpha
        M_s[1, 1] = v.r_alpha**2
        M_s[1, 2] = (v.c - v.a) * v.x_beta + v.r_beta**2
        M_s[2, 0] = v.x_beta
        M_s[2, 1] = (v.c - v.a) * v.x_beta + v.r_beta**2
        M_s[2, 2] = v.r_beta**2
        M_s = mu * M_s

        D_s[0, 0] = sigma * v.zeta_h
        D_s[1, 1] = v.r_alpha**2 * v.zeta_alpha
        D_s[2, 2] = (v.omega_beta / v.omega_alpha) * v.r_beta**2 * v.zeta_beta
        D_s = 2 * mu * D_s

        gamma = 10 # Cubic factor (0 for linear spring)
        K_s[0, 0] = sigma**2 * (1 + gamma * y[3]**2)
        K_s[1, 1] = v.r_alpha**2
        K_s[2, 2] = (v.omega_beta / v.omega_alpha)**2 * v.r_beta**2
        K_s = mu * K_s

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

        M_a = np.zeros((3,3)) # dimensionless aerodynamic inertia matrix
        D_a = np.zeros((3,3)) # dimensionless aerodynamic damping matrix
        K_a = np.zeros((3,3)) # dimensionless aerodynamic stiffness matrix
        L_delta = np.zeros((3, 2)) # dimensionless aero lagging matrix
        Q_a = np.zeros((2, 3)) # matrix of terms multiplied by the modal accelerations
        Q_v = np.zeros((2, 3)) # matrix of terms multiplied by the modal velocities
        L_lambda = np.zeros((2, 2))

        M_a[0, 0] = -1
        M_a[0, 1] = v.a
        M_a[0, 2] = T1 / np.pi
        M_a[1, 0] = v.a
        M_a[1, 1] = -(1/8 + v.a**2)
        M_a[1, 2] = -2 * T13 / np.pi
        M_a[2, 0] = T1 / np.pi
        M_a[2, 1] = -2 * T13 / np.pi
        M_a[2, 2] = T3 / (np.pi**2)

        D_a[0, 0] = (-2)
        D_a[0, 1] = (-2 * (1 - v.a))
        D_a[0, 2] = (T4 - T11) / np.pi
        D_a[1, 0] = (1 + 2 * v.a)
        D_a[1, 1] = (v.a * (1 - 2 * v.a))
        D_a[1, 2] = (T8 - T1 + (v.c - v.a) * T4 + v.a * T11) / np.pi
        D_a[2, 0] = (-T12 / np.pi)
        D_a[2, 1] = (2 * T9 + T1 + (T12 - T4) * (v.a - 0.5)) / np.pi
        D_a[2, 2] = (T11 * (T4 - T12)) / (2 * np.pi**2)
        D_a = V * D_a

        K_a[0, 0] = 0
        K_a[0, 1] = (-2)
        K_a[0, 2] = (-2 * T10) / np.pi
        K_a[1, 0] = 0
        K_a[1, 1] = (1 + 2 * v.a)
        K_a[1, 2] = (2 * v.a * T10 - T4) / np.pi
        K_a[2, 0] = 0
        K_a[2, 1] = (-T12) / np.pi
        K_a[2, 2] = (-1 / np.pi**2) * (T5 - T10 * (T4 - T12))
        K_a = V**2 * K_a
        
        d1 = 0.165
        d2 = 0.335
        # l1 = 0.041
        # l2 = 0.320
        l1 = 0.0455
        l2 = 0.300

        L_delta[0, 0] = d1
        L_delta[0, 1] = d2
        L_delta[1, 0] = - (0.5 + v.a) * d1
        L_delta[1, 1] = - (0.5 + v.a) * d2
        L_delta[2, 0] = T12 * d1 / (2 * np.pi)
        L_delta[2, 1] = T12 * d2 / (2 * np.pi)
        L_delta = 2 * V * L_delta

        Q_a[0, 0] = 1
        Q_a[0, 1] = 0.5 - v.a
        Q_a[0, 2] = T11 / (2 * np.pi)
        Q_a[1, 0] = 1
        Q_a[1, 1] = 0.5 - v.a
        Q_a[1, 2] = T11 / (2 * np.pi)

        Q_v[0, 1] = 1
        Q_v[0, 2] = T10 / np.pi
        Q_v[1, 1] = 1
        Q_v[1, 2] = T10 / np.pi
        Q_v = V * Q_v

        L_lambda[0, 0] = - l1
        L_lambda[1, 1] = - l2
        L_lambda = V * L_lambda

        inv_inert_mat = np.linalg.inv(M_s - M_a)
        A[0:3, 0:3] = - inv_inert_mat @ (D_s - D_a)
        A[0:3, 3:6] = - inv_inert_mat @ (K_s - K_a)
        A[0:3, 6:8] = inv_inert_mat @ L_delta
        A[3:6, 0:3] = np.eye(3)
        A[6:8, 0:3] = Q_a @ A[0:3, 0:3]+ Q_v
        A[6:8, 3:6] = Q_a @ A[0:3, 3:6]
        A[6:8, 6:8] = Q_a @ A[0:3, 6:8] + L_lambda

        return A @ y

    return monolithic_system

def compute_psd(t, data):
    """Compute PSD with consistent parameters"""
    sampling_rate = 1 / np.mean(np.diff(t))
    frequencies, psd = sp.signal.welch(
        data, 
        fs=sampling_rate,
        nperseg=len(t)//2
    )
    mask = frequencies < 1.0
    psd_db = 10 * np.log10(psd)

    # return frequencies[mask], psd_db[mask]
    return frequencies, psd

def add_data_and_psd(fig, time, data, name, row_data, col_data, mode='lines', marker_size=4):
    """Add time series and PSD data to plotly figure"""
    # Add time series data
    fig.add_trace(
        go.Scattergl(
            x=time, 
            y=data, 
            name=name,
            mode=mode,
            marker=dict(size=marker_size) if mode in ['markers', 'lines+markers'] else None,
            showlegend=True
        ),
        row=row_data, 
        col=col_data
    )
    
    # Add PSD data
    frequencies, psd = compute_psd(time, data)
    fig.add_trace(
        go.Scattergl(
            x=frequencies,
            y=psd,
            name=name,
            mode=mode,
            marker=dict(size=marker_size) if mode in ['markers', 'lines+markers'] else None,
            showlegend=False
        ),
        row=row_data+1,
        col=col_data
    )

def format_subplot(fig, row, col, xlabel, ylabel):
    """Format a specific subplot with labels and grid"""
    fig.update_xaxes(
        title_text=xlabel,
        row=row,
        col=col,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
    )
    fig.update_yaxes(
        title_text=ylabel,
        row=row,
        col=col,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
    )

if __name__ == "__main__":
    v = Vars(
        a = -0.5,
        b = 0.127, # m
        c = 0.5,
        C_h = 1.7628, # kg/ms
        C_alpha = 0.0231, # kg/ms
        C_beta = 0.0008, # kg/ms
        I_alpha = 0.01347, # kgm
        I_beta = 0.0003264, # kgm
        k_h = 2818.8, # kg/ms^2
        k_alpha = 37.34, # kg/ms^2
        k_beta = 3.9, # kg/ms^2
        m = 1.558, # kg/m
        m_t = 3.3843, # kg/m
        r_alpha = 0.7321,
        r_beta = 0.1140,
        S_alpha = 0.08587, # kg
        S_beta = 0.00395, # kg
        x_alpha = 0.4340,
        x_beta = 0.02,
        omega_h = 42.5352, # Hz
        omega_alpha = 52.6506, # Hz
        omega_beta = 109.3093,
        rho = 1.225, # kg/m^3
        zeta_h = 0.0133,
        zeta_alpha = 0.01626,
        zeta_beta = 0.0133,
        U = 23.72 # m/s
    )

    dt = 0.02
    t_final = 5 * v.omega_alpha
    vec_t = np.arange(0, t_final, dt)
    n = len(vec_t)

    # hd, ad, bd, h, a, b, x1, x2
    y0 = np.array([
        0, # dh / dt
        0, # dalpha / dt
        0, # dbeta / dt
        0.005 / v.b, # h
        0, # alpha
        0, # beta
        0, # x1
        0  # x2
    ], dtype=np.float64)
    system = create_monolithic_system(y0, v)
    mono = solve_ivp(system, (0, t_final), y0, t_eval=vec_t, method='RK45')
    
    if (not mono.success):
        print("Monolithic solver failed")
        exit(1)

    # Plotting
    fig = make_subplots(
        rows=6, cols=2,
        subplot_titles=(
            'Heave', 'Heave Velocity',
            'Heave PSD', 'Heave Velocity PSD',
            'Wing Pitch', 'Wing Pitch Velocity',
            'Pitch PSD', 'Pitch Velocity PSD',
            'Flap', 'Flap Velocity',
            'Flap PSD', 'Flap Velocity PSD'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )
    add_data_and_psd(fig, vec_t, mono.y[3, :], "Theodorsen", 1, 1)
    add_data_and_psd(fig, vec_t, mono.y[0, :], "Theodorsen", 1, 2)
    add_data_and_psd(fig, vec_t, mono.y[4, :], "Theodorsen", 3, 1)
    add_data_and_psd(fig, vec_t, mono.y[1, :], "Theodorsen", 3, 2)
    add_data_and_psd(fig, vec_t, mono.y[5, :], "Theodorsen", 5, 1)
    add_data_and_psd(fig, vec_t, mono.y[2, :], "Theodorsen", 5, 2)

    format_subplot(fig, 1, 1, "t", r"y")
    format_subplot(fig, 1, 2, "t", r"$\dot{y}$")
    format_subplot(fig, 2, 1, "f", "Amplitude (dB)")
    format_subplot(fig, 2, 2, "f", "Amplitude (dB)")
    format_subplot(fig, 3, 1, "t", r"$\alpha$")
    format_subplot(fig, 3, 2, "t", r"$\dot{\alpha}$")
    format_subplot(fig, 4, 1, "f", "Amplitude (dB)")
    format_subplot(fig, 4, 2, "f", "Amplitude (dB)")
    format_subplot(fig, 5, 1, "t", r"$\beta$")
    format_subplot(fig, 5, 2, "t", r"$\dot{\beta}$")
    format_subplot(fig, 6, 1, "f", "Amplitude (dB)")
    format_subplot(fig, 6, 2, "f", "Amplitude (dB)")

    fig.update_layout(
        title="Theodorsen 3DOF Aeroelastic Response",
        title_x=0.5,
        autosize=True,
        showlegend=True,
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.0
        )
    )

    fig.write_html("build/3dof.html", include_mathjax='cdn')

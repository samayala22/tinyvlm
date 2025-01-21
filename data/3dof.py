import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
import scipy as sp

@dataclass
class Vars:
    b: float # half chord length
    a: float # nd elastic axis position relative to semi chord
    rho: float # fluid density (kg/m^3)
    m_w: float # mass of the wing (kg)
    m_t: float # mass of wing + flap (kg)
    zeta_alpha: float # wing pitch damping ratio
    zeta_y: float # wing heave damping ratio
    zeta_beta: float # flap pitch damping ratio
    k_alpha: float # wing pitch stiffness (N/m)
    k_y: float # wing heave stiffness (N/m)
    k_beta: float # flap pitch stiffness (N/m)
    x_alpha: float # distance between wing cg relative to elastic axis
    x_beta: float # distance between flap cg relative to elastic axis
    r_alpha: float # wing radius of gyration
    r_beta: float # flap radius of gyration
    U: float # velocity

def newmark_beta_step(M, C, K, u_i, v_i, a_i, delta_F, dt, beta=1/4, gamma=1/2):
    """
    Implicit Newmark-Beta Method for Structural Dynamics.

    Parameters:
    - M, C, K: Mass, Damping, and Stiffness matrices (n x n).
    - beta, gamma: Newmark parameters.

    Returns: incremental variation for displacement, velocity, and acceleration.
    """

    # Precompute constants
    x2 = 1
    x1 = gamma / (beta * dt)
    x0 = 1 / (beta * dt**2)
    xd0 = 1 / (beta * dt)
    xd1 = gamma / beta
    xdd0 = 1/(2*beta)
    xdd1 = - dt * (1 - gamma / (2*beta))

    # Effective stiffness matrix
    K_eff = x0 * M + x1 * C + x2 * K
    F_eff = delta_F + M @ (xd0 * v_i + xdd0 * a_i) + C @ (xd1 * v_i + xdd1 * a_i)
    du = np.linalg.solve(K_eff, F_eff)
    dv = x1 * du - xd1 * v_i - xdd1 * a_i
    da = x0 * du - xd0 * v_i - xdd0 * a_i

    return du, dv, da

def monolithic_system_3dof(v: Vars):
    """
    Creates the M, C, K matrices of the coupled 3dof aeroelastic system
    q = [y, alpha, beta, x] where x is the augmented variable    
    $$
    M =\left[\begin{array}{cccc}
    m_T+\pi \rho b^2 & S_\alpha-a \pi \rho b^3 & S_\beta-\rho b^3 T_1 & 0 \\
    S_\alpha-a \pi \rho b^3 & I_\alpha+\pi\left(\frac{1}{8}+a^2\right) \rho b^4 & I_\beta+S_\beta(c-a) b-\rho b^4\left[T_7+(c-a) T_1\right] & 0 \\
    S_\beta-\rho b^3 T_1 & I_\beta+S_\beta(c-a) b+2 \rho b^4 T_{13} & I_\beta-\frac{\rho b^4}{\pi} T_3 & 0 \\
    0 & 0 & 0 & 1
    \end{array}\right]
    $$
    $$
    C =\left[\begin{array}{ccc}
    d_y+2 \pi \rho b U\left(c_0-c_1-c_3\right) & \left(1+\left(c_0-c_1-c_3\right)(1-2 a)\right) \pi \rho b^2 U & \left(T_{11}\left(c_0-c_1-c_3\right)-T_4\right) \rho U b^2 & 2 \pi \rho U^2 b\left(c_1 c_2+c_3 c_4\right)\\
    -2 \pi \rho b^2 U\left(a+\frac{1}{2}\right)\left(c_0-c_1-c_3\right) & d_\alpha+\left(\frac{1}{2}-a\right)\left(1-\left(c_0-c_1-c_3\right)(1+2 a)\right) \pi \rho b^3 U & {\left[T_1-T_8-(c-a) T_4+\frac{T_{11}}{2}-(2 a+1) T_{11}\left(c_0-c_1-c_3\right)\right] \rho b^3 U} & {-2 \pi \rho U^2 b^2\left(a+\frac{1}{2}\right)\left(c_1 c_2+c_3 c_4\right)} \\
    \rho U b^2 T_{12}\left(c_0-c_1-c_3\right) & {\left[T_4\left(a-\frac{1}{2}\right)-T_1-2 T_9+T_{12}\left(\frac{1}{2}-a\right)\left(c_0-c_1-c_3\right)\right] \rho U b^3} & d_\beta+\left[T_{11} T_{12}\left(c_0-c_1-c_3\right)-T_4 T_{11}\right] \frac{\rho U b^3}{2 \pi} & {\rho U^2 b^2 T_{12}\left(c_1 c_2+c_3 c_4\right)} \\
    -\frac{1}{b} & a-\frac{1}{2} & {-\frac{T_{11}}{2 \pi}} & {\left(c_2+c_4\right) \frac{U}{b}}
    \end{array}\right]
    $$
    $$
    K =\left[\begin{array}{cccc}
    k_{y 0} & 2 \pi \rho U^2 b\left(c_0-c_1-c_3\right) & 2 \rho U^2 b T_{10}\left(c_0-c_1-c_3\right) & 2 \pi \rho U^3\left(c_2 c_4\left(c_1+c_3\right)\right) \\
    0 & k_{\alpha 0}-2 \pi\left(\frac{1}{2}+a\right)\left(c_0-c_1-c_3\right) \rho b^2 U^2 & \rho U^2 b^2\left[T_4+T_{10}+T_{10}(2 a+1)\left(c_0-c_1-c_3\right)\right] & -2 \pi \rho U^3 b\left(a+\frac{1}{2}\right)\left(c_2 c_4\left(c_1+c_3\right)\right) \\
    0 & \rho U^2 b^2 T_{12}\left(c_0-c_1-c_3\right) & k_{\beta 0}+\left(T_5-T_4 T_{10}+T_{10} T_{12}\left(c_0-c_1-c_3\right)\right) \frac{\rho U^2 b^2}{\pi} & \rho U^3 b T_{12}\left(c_2 c_4\left(c_1+c_3\right)\right) \\
    0 & -\frac{U}{b} & -\frac{T_{10} U}{b \pi} & c_2 c_4 \frac{U^2}{b^2}
    \end{array}\right]
    $$


    """
    S_alpha = v.x_alpha * v.m_w * v.b
    S_beta = v.x_beta * v.m_w * v.b
    I_alpha = v.r_alpha**2 * v.m_w * v.b
    I_beta = v.r_beta**2 * v.m_w * v.b

    # Note unsure of these equations
    d_y = 2*v.zeta_y*np.sqrt(v.m_t*v.k_y)
    d_alpha = 2*v.zeta_alpha*np.sqrt(v.m_w*v.k_alpha)
    d_beta = 2*v.zeta_beta*np.sqrt((v.m_t-v.m_w)*v.k_alpha)

    c = 0.5

    sqrt_1_minus_c2 = np.sqrt(1 - c**2)
    acos_c = np.arccos(c)
    p  = (-1/3) * (sqrt_1_minus_c2)**3

    T1 = (-1/3) * sqrt_1_minus_c2 * (2 + c**2) + c * acos_c
    T2 = c * (1 - c**2) - sqrt_1_minus_c2 * (1 + c**2) * acos_c + c * (acos_c)**2
    T3 = -((1/8) + c**2) * (acos_c)**2 + (1/4) * c * sqrt_1_minus_c2 * acos_c * (7 + 2 * c) - (1/8) * (1 - c**2) * (5 * c**2 + 4)
    T4 = -acos_c + c * sqrt_1_minus_c2
    T5 = -(1 - c**2) - (acos_c)**2 + 2 * c * sqrt_1_minus_c2 * acos_c
    T6 = T2
    T7 = -((1/8) + c**2) * acos_c + (1/8) * c * sqrt_1_minus_c2 * (7 + 2 * c**2)
    T8 = (-1/3) * sqrt_1_minus_c2 * (2 * c**2 + 1) + c * acos_c
    T9 = (1/2) * (p + v.a * T4)
    T10 = sqrt_1_minus_c2 + acos_c
    T11 = acos_c * (1 - 2 * c) + sqrt_1_minus_c2 * (2 - c)
    T12 = sqrt_1_minus_c2 * (2 + c) - acos_c * (2 * c + 1)
    T13 = (1/2) * (-T7 - (c - v.a) * T1)
    T14 = (1/16) + (1/2) * v.a * c

    c0 = 1.0
    c1 = 0.165
    c2 = 0.0455
    c3 = 0.335
    c4 = 0.3

    M = np.zeros((4,4), dtype=np.float64)
    C = np.zeros((4,4), dtype=np.float64)
    K = np.zeros((4,4), dtype=np.float64)

    M[0, 0] = v.m_t + np.pi * v.rho * v.b**2
    M[0, 1] = S_alpha - v.a * np.pi * v.rho * v.b**3
    M[0, 2] = S_beta - v.rho * v.b**3 * T1
    M[0, 3] = 0

    M[1, 0] = S_alpha - v.a * np.pi * v.rho * v.b**3
    M[1, 1] = I_alpha + np.pi * (1/8 + v.a**2) * v.rho * v.b**4
    M[1, 2] = I_beta + S_beta * (c - v.a) * v.b - v.rho * v.b**4 * (T7 + (c - v.a) * T1)
    M[1, 3] = 0

    M[2, 0] = S_beta - v.rho * v.b**3 * T1
    M[2, 1] = I_beta + S_beta * (c - v.a) * v.b + 2 * v.rho * v.b**4 * T13
    M[2, 2] = I_beta - (v.rho * v.b**4 / np.pi) * T3
    M[2, 3] = 0

    M[3, 0] = 0
    M[3, 1] = 0
    M[3, 2] = 0
    M[3, 3] = 1

    C[0, 0] = d_y + 2 * np.pi * v.rho * v.b * v.U * (c0 - c1 - c3)
    C[0, 1] = (1 + (c0 - c1 - c3) * (1 - 2 * v.a)) * np.pi * v.rho * v.b**2 * v.U
    C[0, 2] = (T11 * (c0 - c1 - c3) - T4) * v.rho * v.U * v.b**2
    C[0, 3] = 2 * np.pi * v.rho * v.U**2 * v.b * (c1 * c2 + c3 * c4)

    C[1, 0] = -2 * np.pi * v.rho * v.b**2 * v.U * (v.a + 0.5) * (c0 - c1 - c3)
    C[1, 1] = d_alpha + (0.5 - v.a) * (1 - (c0 - c1 - c3) * (1 + 2 * v.a)) * np.pi * v.rho * v.b**3 * v.U
    C[1, 2] = (T1 - T8 - (c - v.a) * T4 + T11 / 2 - (2 * v.a + 1) * T11 * (c0 - c1 - c3)) * v.rho * v.b**3 * v.U
    C[1, 3] = -2 * np.pi * v.rho * v.U**2 * v.b**2 * (v.a + 0.5) * (c1 * c2 + c3 * c4)

    C[2, 0] = v.rho * v.U * v.b**2 * T12 * (c0 - c1 - c3)
    C[2, 1] = (T4 * (v.a - 0.5) - T1 - 2 * T9 + T12 * (0.5 - v.a) * (c0 - c1 - c3)) * v.rho * v.U * v.b**3
    C[2, 2] = d_beta + (T11 * T12 * (c0 - c1 - c3) - T4 * T11) * (v.rho * v.U * v.b**3) / (2 * np.pi)
    C[2, 3] = v.rho * v.U**2 * v.b**2 * T12 * (c1 * c2 + c3 * c4)

    C[3, 0] = -1 / v.b
    C[3, 1] = v.a - 0.5
    C[3, 2] = -T11 / (2 * np.pi)
    C[3, 3] = (c2 + c4) * v.U / v.b

    K[0, 0] = v.k_y
    K[0, 1] = 2 * np.pi * v.rho * v.U**2 * v.b * (c0 - c1 - c3)
    K[0, 2] = 2 * v.rho * v.U**2 * v.b * T10 * (c0 - c1 - c3)
    K[0, 3] = 2 * np.pi * v.rho * v.U**3 * (c2 * c4 * (c1 + c3))

    K[1, 0] = 0
    K[1, 1] = v.k_alpha - 2 * np.pi * (0.5 + v.a) * (c0 - c1 - c3) * v.rho * v.b**2 * v.U**2
    K[1, 2] = v.rho * v.U**2 * v.b**2 * (T4 + T10 + T10 * (2 * v.a + 1) * (c0 - c1 - c3))
    K[1, 3] = -2 * np.pi * v.rho * v.U**3 * v.b * (v.a + 0.5) * (c2 * c4 * (c1 + c3))

    K[2, 0] = 0
    K[2, 1] = v.rho * v.U**2 * v.b**2 * T12 * (c0 - c1 - c3)
    K[2, 2] = v.k_beta + (T5 - T4 * T10 + T10 * T12 * (c0 - c1 - c3)) * (v.rho * v.U**2 * v.b**2) / np.pi
    K[2, 3] = v.rho * v.U**3 * v.b * T12 * (c2 * c4 * (c1 + c3))

    K[3, 0] = 0
    K[3, 1] = -v.U / v.b
    K[3, 2] = -T10 * v.U / (v.b * np.pi)
    K[3, 3] = c2 * c4 * v.U**2 / v.b**2

    return M, C, K

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
        b = 0.127,
        a = -0.5,
        rho = 1.225,
        m_w = 1.7025,
        m_t = 3.5021,
        zeta_alpha = 0.01,
        zeta_y = 0.012,
        zeta_beta = 0.015,
        k_alpha = 1114.0,
        k_y = 17240,
        k_beta = 73.07,
        x_alpha = 0.3294,
        x_beta = 0.01795,
        r_alpha = 0.6840,
        r_beta = 0.07336,
        U = 22.0
    )

    dt = 0.01
    t_final = 5.0
    vec_t = np.arange(0, t_final, dt)
    n = len(vec_t)

    M, C, K = monolithic_system_3dof(v)

    # Historical data
    u = np.zeros((4, n), dtype=np.float64)
    v = np.zeros((4, n), dtype=np.float64)
    a = np.zeros((4, n), dtype=np.float64)
    F = np.zeros((4, n), dtype=np.float64) # external forces / nonlinear terms
    dF = np.zeros(4, dtype=np.float64)

    # Initial conditions
    u[:, 0] = np.array([0, np.radians(3), 0, 0], dtype=np.float64)
    v[:, 0] = np.array([0, 0, 0, 0], dtype=np.float64)
    a[:, 0] = np.linalg.solve(M, F[:, 0] - C @ v[:,0] - K @ u[:,0])
    
    for i in range(n-1):
        du, dv, da = newmark_beta_step(M, C, K, u[:,i], v[:,i], a[:,i], dF, dt)
        u[:, i+1] = u[:, i] + du
        v[:, i+1] = v[:, i] + dv
        a[:, i+1] = a[:, i] + da

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
    add_data_and_psd(fig, vec_t, u[0, :], "Theodorsen", 1, 1)
    add_data_and_psd(fig, vec_t, v[0, :], "Theodorsen", 1, 2)
    add_data_and_psd(fig, vec_t, np.degrees(u[1, :]), "Theodorsen", 3, 1)
    add_data_and_psd(fig, vec_t, np.degrees(v[1, :]), "Theodorsen", 3, 2)
    add_data_and_psd(fig, vec_t, np.degrees(u[2, :]), "Theodorsen", 5, 1)
    add_data_and_psd(fig, vec_t, np.degrees(v[2, :]), "Theodorsen", 5, 2)

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

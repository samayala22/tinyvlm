import numpy as np
from pathlib import Path
import plotly.graph_objects as go

import scipy as sp
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from tqdm import tqdm
import plotting as plot

# Local imports
import integrators

EPS_sqrt_f = np.sqrt(1.19209e-07)

# Dimensional variables
class DVars:
    def __init__(self, rho, u_inf, b, p_axis, x_a, r_a, m, k_h, k_a):
        # Independent
        self.rho = rho # fluid density
        self.u_inf = u_inf # freestream velocity
        self.b = b # half chord
        self.p_axis = p_axis # position of elastic / pitch axis (measured from center of wing)
        self.x_a = x_a # mass center (from pitch axis to center of mass)
        self.r_a = r_a # radius of gyration around elastic axis
        self.m = m # mass
        self.k_h = k_h # linear stiffness
        self.k_a = k_a # torsional stiffness
        # Dependent
        self.c = 2*b
        self.S_a = m * x_a # first moment of inertia
        self.I_a = m * x_a**2 # second moment of inertia

class NDVars:
    def from_dvars(self, d_vars):
        self.U = d_vars.u_inf / (self.omega_a * d_vars.b) # reduced velocity
        self.mu = d_vars.m / (np.pi * d_vars.rho * d_vars.b**2) # mass ratio
        self.r_a = d_vars.r_a / d_vars.b # dimensionless radius of gyration
        self.x_a = d_vars.x_a / d_vars.b # dimensionless mass center

    def __init__(self, a_h, omega, zeta_a, zeta_h,x_a, mu, r_a, U):
        self.a_h = a_h # dimensionless pitch axis
        self.omega = omega # dimensionless ratio of torsional stiffness to linear stiffness
        self.zeta_a = zeta_a # dimensionless pitch axis
        self.zeta_h = zeta_h # dimensionless pitch axis
        self.x_a = x_a # dimensionless mass center
        self.mu = mu # mass ratio
        self.r_a = r_a # dimensionless radius of gyration
        self.U = U # reduced velocity

def compute_logarithmic_decrement(signal):
    peaks_idx, _ = find_peaks(signal)
    num_peaks = len(peaks_idx)
    peak_values = signal[peaks_idx]
    
    decrements = np.log((peak_values[:-1]+EPS_sqrt_f) / (peak_values[1:num_peaks+1]+EPS_sqrt_f))
    delta = np.mean(decrements)
    return delta

def compute_damping_ratio(log_decrement):
    return log_decrement / np.sqrt(4 * np.pi**2 + log_decrement**2)

def get_damping_ratio(signal):
    delta = compute_logarithmic_decrement(signal)
    zeta = compute_damping_ratio(delta)
    return zeta

def get_peaks(time, signal):
    peaks_idx, _ = find_peaks(signal)
    return time[peaks_idx], signal[peaks_idx]

def damped_oscillation(t, A, gamma, omega, phi, C):
    """
    Damped or amplified oscillatory function.

    Parameters:
        t (np.ndarray): Time vector.
        A (float): Amplitude.
        gamma (float): Growth (positive) or damping (negative) rate.
        omega (float): Angular frequency.
        phi (float): Phase shift.
        C (float): Constant offset.

    Returns:
        np.ndarray: Oscillatory signal at time t.
    """
    return A * np.exp(gamma * t) * np.sin(omega * t + phi) + C

def alpha_freeplay(alpha, M0 = 0.0, Mf = 0.0, delta = np.radians(0.5), a_f = np.radians(0.25)):
    if (alpha < a_f):
        return M0 + alpha - a_f
    elif (alpha >= a_f and alpha <= (a_f + delta)):
        return M0 + Mf * (alpha - a_f)
    else: # alpha > a_F + delta
        return M0 + alpha - a_f + delta * (Mf - 1)

def alpha_cubic(alpha, beta0 = 0.0, beta1 = 0.1, beta2 = 0.0, beta3 = 40.0):
    return beta0 + beta1*alpha + beta2*alpha**2 + beta3*alpha**3

def alpha_linear(alpha):
    return alpha

def create_monolithic_system(y0: np.ndarray, ndv: NDVars, M: callable):
    psi1 = 0.165
    psi2 = 0.335
    eps1 = 0.0455
    eps2 = 0.3

    M1 = np.zeros((6,6))
    M2 = np.zeros((6,6))

    M2[0, 0] = 1
    M2[1, 1] = 1
    M2[2, 2] = 1 + 1/ndv.mu
    M2[2, 3] = ndv.x_a - ndv.a_h/ndv.mu
    M2[3, 2] = ndv.x_a / (ndv.r_a**2) - ndv.a_h/(ndv.mu*ndv.r_a**2)
    M2[3, 3] = 1.0 + (2/(ndv.mu*ndv.r_a**2))*((ndv.a_h**2)/2 + 1/16)
    M2[4, 2] = -1
    M2[4, 3] = - (0.5 - ndv.a_h)
    M2[4, 4] = 1
    M2[5, 2] = -1
    M2[5, 3] = - (0.5 - ndv.a_h)
    M2[5, 5] = 1

    f0 = 0.5 - ndv.a_h
    f1 = - 1 / (np.pi * ndv.mu)
    f2 = 2*np.pi
    f3 = 2 / (np.pi * ndv.mu * ndv.r_a**2)
    f4 = np.pi * (0.5 + ndv.a_h)

    M1[0, 2] = 1
    M1[1, 3] = 1
    M1[2, 0] = - (ndv.omega/ndv.U)**2
    M1[2, 1] = - 2 / ndv.mu
    M1[2, 2] = - 2.0*ndv.zeta_h*(ndv.omega/ndv.U) - 2 / ndv.mu
    M1[2, 3] = - (1/(np.pi*ndv.mu)) * (np.pi + 2*np.pi*(0.5 - ndv.a_h))
    M1[2, 4] = (2 / ndv.mu) * psi1
    M1[2, 5] = (2 / ndv.mu) * psi2
    M1[3, 1] = - 1/(ndv.U**2) + f3*f4
    # M1[3, 1] = f3*f4
    M1[3, 2] = f3*f4
    M1[3, 3] = - 2.0*ndv.zeta_a/ndv.U + f3*f4*f0 - f3*0.5*np.pi*f0
    M1[3, 4] = - psi1*f3*f4
    M1[3, 5] = - psi2*f3*f4
    M1[4, 3] = 1
    M1[4, 4] = -eps1
    M1[5, 3] = 1
    M1[5, 5] = -eps2

    A = np.linalg.solve(M2, M1)
    eigvalues, eigvectors = np.linalg.eig(A)

    # linear stiffness moved to V
    M1[2, 0] = 0.0
    M1[3, 1] = f3*f4

    def monolithic_system(t, y: np.ndarray):
        # Yung p212
        """
        system unknowns:
        q = [h, a, hd, ad, x1, x2]
        """
        f5 = (y0[1] + y0[2] + f0*y0[3])*(-psi1*np.exp(-eps1*t) - psi2*np.exp(-eps2*t))

        V = np.zeros(6)

        V[2] = f1*f2*f5
        V[3] = f3*f4*f5
            
        V[2] += - (ndv.omega/ndv.U)**2 * y[0]
        V[3] += - 1/(ndv.U**2) * M(y[1])

        y_d = np.linalg.solve(M2, M1 @ y + V)

        return y_d
    
    return monolithic_system, eigvalues

def solve_iterative(ndv: NDVars, t_final_nd, dt_nd):
    vec_t_nd = np.arange(0, t_final_nd, dt_nd)
    n = len(vec_t_nd)

    m = ndv.mu * (np.pi * rho * b**2)
    omega_a = np.sqrt(k_a / (m * (ndv.r_a * b)**2))
    U = U_vel * (b * omega_a)
    omega_h = ndv.omega * omega_a
    k_h = omega_h**2 * m

    # Aeroelastic equations of motion mass, damping and stiffness matrices
    M = np.array([
        [1.0, ndv.x_a],
        # [1.0, 0.0],
        [ndv.x_a / (ndv.r_a**2), 1.0]
    ])
    C = np.array([
        [2.0*ndv.zeta_h*(ndv.omega/ndv.U), 0],
        [0, 2.0*ndv.zeta_a/ndv.U]
    ])
    # K = np.array([
    #     [(ndv.omega/ndv.U)**2, 0],
    #     [0, 1/(ndv.U**2)]
    # ])
    K = np.zeros((2,2)) # linear terms moved to the rhs

    # Time history of dofs
    u = np.zeros((2, len(vec_t_nd)))
    v = np.zeros((2, len(vec_t_nd)))
    a = np.zeros((2, len(vec_t_nd)))
    F = np.zeros((2, len(vec_t_nd)))

    # Augmented aero states
    vec_w1 = np.zeros(len(vec_t_nd))
    vec_w2 = np.zeros(len(vec_t_nd))

    # Initial condition
    u[:, 0] = np.array([0.0, np.radians(3)])
    v[:, 0] = np.array([0, 0])
    # a[:, 0] = np.linalg.solve(M, F[:, 0] - C @ v[:,0] - K @ u[:,0])
    init_R = np.array([
        - (ndv.omega / ndv.U)**2 * u[0,0],
        - 1/(ndv.U**2) * torsional_func(u[1,0])
    ])
    a[:, 0] = np.linalg.solve(M, F[:, 0] + init_R - C @ v[:,0] - K @ u[:,0])
    def w(s: float):
        idx = int(s / dt_nd)
        return v[1, idx] + a[0, idx] + (0.5 - ndv.a_h) * a[1, idx]

    def dw1ds(s: float, w1: float): return w(s) - eps1 * w1
    def dw2ds(s: float, w2: float): return w(s) - eps2 * w2

    def aero(i):
        t = vec_t_nd[i]

        integrators.rk2_step(dw1ds, vec_w1, vec_t_nd[i-1], dt_nd, i-1)
        integrators.rk2_step(dw2ds, vec_w2, vec_t_nd[i-1], dt_nd, i-1)
        w1 = vec_w1[i]
        w2 = vec_w2[i]

        duhamel = u[1, i] - u[1, 0] + v[0, i] - v[0, 0] + (0.5 - ndv.a_h)*(v[1, i] - v[1, 0]) - psi1*w1 - psi2*w2
        wagner_init = (u[1, 0] + u[0, 0] + (0.5 - ndv.a_h)*v[1,0])*(1 - psi1*np.exp(-eps1*t) - psi2*np.exp(-eps2*t))
        cl = np.pi*(a[0, i] - ndv.a_h * a[1, i] + v[1, i]) + 2*np.pi*(wagner_init + duhamel)
        cm = np.pi*(0.5 + ndv.a_h)*(wagner_init + duhamel) + 0.5*np.pi*ndv.a_h*(a[0, i] - ndv.a_h*a[1, i]) - 0.5*np.pi*(0.5 - ndv.a_h)*v[1, i] - (np.pi/16) * a[1, i]
        # print(f"i: {i} | cl: {cl} | cm: {cm}")

        return np.array([
            - cl / (np.pi*ndv.mu),
            (2*cm) / (np.pi*ndv.mu*ndv.r_a**2)
        ])

    # Newmark V2
    max_iter = 50
    for i in range(n-1):
        t = vec_t_nd[i]
        delta_F = np.zeros(2)
        du = np.zeros(2)
        du_k = np.zeros(2) + 1
        iteration = 0
        # Newton-Raphson iterations
        while (np.linalg.norm(du_k - du) / len(du) > newton_err_thresh):
            du_k = du[:]
            du, dv, da = integrators.newmark_beta_step(M, C, K, v[:,i], a[:,i], delta_F, dt_nd)

            u[:,i+1] = u[:,i] + du
            v[:,i+1] = v[:,i] + dv
            a[:,i+1] = a[:,i] + da

            F[:,i+1] = aero(i+1)
            delta_F = F[:,i+1] - F[:,i]
            delta_F[0] += - (ndv.omega / ndv.U)**2 * du[0]
            delta_F[1] += - 1/(ndv.U**2) * (torsional_func(u[1,i+1]) - torsional_func(u[1,i]))

            iteration += 1
            if (iteration > max_iter):
                print("Newton process did not converge")
                break
        # print("iters: ", iteration)
    
    return vec_t_nd, u, v, a, F

def plot_uvlm(fig):
    uvlm_t = []
    uvlm_h = []
    uvlm_a = []
    uvlm_hd = []
    uvlm_ad = []
    uvlm_f_h = []
    uvlm_f_a = []
    with open("build/windows/x64/release/2dof.txt", "r") as f:
        f.readline() # skip first line
        for line in f:
            t, h, a, hd, ad, f_h, f_a = map(float, line.split())
            uvlm_t.append(t)
            uvlm_h.append(h)
            uvlm_a.append(a)
            uvlm_hd.append(hd)
            uvlm_ad.append(ad)
            uvlm_f_h.append(f_h)
            uvlm_f_a.append(f_a)

    uvlm_t = np.array(uvlm_t)
    uvlm_h = np.array(uvlm_h)
    uvlm_a = np.array(uvlm_a)
    uvlm_hd = np.array(uvlm_hd)
    uvlm_ad = np.array(uvlm_ad)
    uvlm_f_h = np.array(uvlm_f_h)
    uvlm_f_a = np.array(uvlm_f_a)

    plot.add_data_and_psd(fig, uvlm_t, uvlm_h, "UVLM", 1, 1, 0, mode='markers')
    plot.add_data_and_psd(fig, uvlm_t, uvlm_hd, "UVLM", 1, 2, 0, mode='markers')
    plot.add_data_and_psd(fig, uvlm_t, uvlm_f_h, "UVLM", 1, 3, 0, mode='markers')
    plot.add_data_and_psd(fig, uvlm_t, uvlm_a, "UVLM", 3, 1, 0, mode='markers')
    plot.add_data_and_psd(fig, uvlm_t, uvlm_ad, "UVLM", 3, 2, 0, mode='markers')
    plot.add_data_and_psd(fig, uvlm_t, uvlm_f_a, "UVLM", 3, 3, 0, mode='markers')

def plot_monolithic(fig, monolithic_sol):
    plot.add_data_and_psd(fig, monolithic_sol.t, monolithic_sol.y[0, :], "Theodorsen Monolithic", 1, 1, 1)
    plot.add_data_and_psd(fig, monolithic_sol.t, monolithic_sol.y[2, :], "Theodorsen Monolithic", 1, 2, 1)
    plot.add_data_and_psd(fig, monolithic_sol.t, monolithic_sol.y[1, :], "Theodorsen Monolithic", 3, 1, 1)
    plot.add_data_and_psd(fig, monolithic_sol.t, monolithic_sol.y[3, :], "Theodorsen Monolithic", 3, 2, 1)

def plot_iterative(fig):
    vec_t_nd, u, v, a, F = solve_iterative(ndv, t_final_nd, dt_nd)

    plot.add_data_and_psd(fig, vec_t_nd, u[0, :], "Theodorsen Iterative", 1, 1, 2)
    plot.add_data_and_psd(fig, vec_t_nd, v[0, :], "Theodorsen Iterative", 1, 2, 2)
    plot.add_data_and_psd(fig, vec_t_nd, F[0, :], "Theodorsen Iterative", 1, 3, 2)
    plot.add_data_and_psd(fig, vec_t_nd, u[1, :], "Theodorsen Iterative", 3, 1, 2)
    plot.add_data_and_psd(fig, vec_t_nd, v[1, :], "Theodorsen Iterative", 3, 2, 2)
    plot.add_data_and_psd(fig, vec_t_nd, F[1, :], "Theodorsen Iterative", 3, 3, 2)

def format_fig(fig):
    # Time series plots - Row 1
    plot.format_subplot(fig, 1, 1, r"$\bar{t}$", r"$\bar{h}$")
    plot.format_subplot(fig, 1, 2, r"$\bar{t}$", r"$\bar{\dot{h}}$")
    plot.format_subplot(fig, 1, 3, r"$\bar{t}$", r"$F_{h}$")
    
    # PSD plots - Row 2
    plot.format_subplot(fig, 2, 1, r"$\bar{f}$", "Amplitude")
    plot.format_subplot(fig, 2, 2, r"$\bar{f}$", "Amplitude")
    plot.format_subplot(fig, 2, 3, r"$\bar{f}$", "Amplitude")
    
    # Time series plots - Row 3
    plot.format_subplot(fig, 3, 1, r"$\bar{t}$", r"$\alpha$ (deg)")
    plot.format_subplot(fig, 3, 2, r"$\bar{t}$", r"$\dot{\alpha}$")
    plot.format_subplot(fig, 3, 3, r"$\bar{t}$", r"$F_{\alpha}$")
    
    # PSD plots - Row 4
    plot.format_subplot(fig, 4, 1, r"$\bar{f}$", "Amplitude")
    plot.format_subplot(fig, 4, 2, r"$\bar{f}$", "Amplitude")
    plot.format_subplot(fig, 4, 3, r"$\bar{f}$", "Amplitude")

if __name__ == "__main__":
    # TODO: move these coefficients elsewhere
    psi1 = 0.165
    psi2 = 0.335
    eps1 = 0.0455
    eps2 = 0.3
    dofs = 2
    # Dimensionless params
    flutter_speed = 6.285
    flutter_ratio = 0.6
    # vec_U = np.linspace(0.1, 5.5, 200) # for freeplay
    # vec_U = np.linspace(0.5, 7.0, 200) # for cubic
    # vec_U = [flutter_ratio * flutter_speed] # reduced velocity
    # vec_U = [2.0] # reduced velocity
    vec_U = [3.5]
    newton_err_thresh = 1e-8
    torsional_spring = 0
    torsional_spring_names = ["Freeplay", "Cubic", "Linear"]
    peaks_data = [[], []]
    peaks_U = [[], []]
    eigenvalues = np.zeros((2*dofs, len(vec_U)), dtype=np.complex128)

    if (torsional_spring == 0):
        torsional_func = alpha_freeplay
    elif (torsional_spring == 1):
        torsional_func = alpha_cubic
    else:
        torsional_func = alpha_linear

    # Dimensional parameters
    b = 0.5
    k_a = 1000.0
    rho = 1.0

    # Storage for multiple simulations
    freqs = np.zeros((2, len(vec_U)))
    damping_ratios = np.zeros((2, len(vec_U)))
    damping_ratios_m = np.zeros((2, len(vec_U)))

    # for idx, U_vel in enumerate(vec_U):
    for idx, U_vel in tqdm(enumerate(vec_U), total=len(vec_U)):
        # Zhao 2004 params
        ndv = NDVars(
            a_h = -0.5,
            omega = 0.2,
            zeta_a = 0.0,
            zeta_h = 0.0,
            x_a = 0.25,
            mu = 100.0,
            r_a = 0.5,
            U = U_vel
        )

        # Dimensionless parameters
        dt_nd = 0.2
        t_final_nd = 1000.0

        y0 = np.array([0, np.radians(3), 0, 0, 0, 0]) # h, a, hd, ad, x1, x2
        system, eigvalues = create_monolithic_system(y0, ndv, torsional_func)
        monolithic_sol = solve_ivp(system, (0, t_final_nd), y0, t_eval=np.arange(0, t_final_nd, dt_nd), method='RK45')

        max_real_part = np.max(eigvalues.real)
        # if max_real_part > 0:
        #     print("Linear flutter detected at U =", U_vel)

        sorted_eigvals = np.sort_complex(eigvalues)[::-1]

        eigenvalues[:, idx] = sorted_eigvals[:eigenvalues.shape[0]]

        if (len(vec_U) == 1):
            fig = plot.create_dofs_figure(["Heave", "Pitch"])
            # plot_uvlm(fig)
            plot_monolithic(fig, monolithic_sol)
            plot_iterative(fig)
            format_fig(fig)

            # param_str = f"{ndv.U:.1f}".replace('.', '_')
            # plot.fig_save(fig, f"build/2dof/2dof_{torsional_spring_names[torsional_spring]}_{param_str}")
            plot.fig_save(fig, f"build/2dof/2dof", pdf=False)

            # Poincare sections
            start = int(0.9*len(monolithic_sol.t))
            fig2 = plot.fig_create(2, 1, ("Heave", "Pitch"))
            for i in range(2):
                fig2.add_trace(
                    go.Scatter(
                        x=monolithic_sol.y[i, start:], 
                        y=monolithic_sol.y[i + 2, start:], 
                        mode="markers"
                    ),
                    row=i + 1, 
                    col=1
                )
            plot.format_subplot(fig2, 1, 1, r"$\bar{h}$", r"$\dot{\bar{h}}$")
            plot.format_subplot(fig2, 2, 1, r"$\alpha$", r"$\dot{\alpha}$")
            plot.fig_save(fig2, f"build/2dof/2dof_{torsional_spring_names[torsional_spring]}_poincare", pdf=False)

        else:
            slice_start = int(0.75 * len(monolithic_sol.t))
            peaks_slice = monolithic_sol.y[0:2, slice_start:]
            
            for i in range(2):
                peaks_data[i].append(peaks_slice[i, plot.find_peak_idx(peaks_slice[i, :])])
                peaks_U[i].append(np.array([U_vel] * peaks_data[i][-1].shape[0]))

    if (len(vec_U) > 1):
        fig = plot.fig_create(2, 1, ("Heave", "Pitch"))
        for i in range(2):
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate(peaks_U[i]), 
                    y=np.concatenate(peaks_data[i]), 
                    mode="markers"
                ),
                row=i + 1, 
                col=1
            )
            plot.format_subplot(fig, i + 1, 1, r"$\bar{U}$", "Amplitude")
        
        plot.fig_save(fig, f"build/2dof/2dof_{torsional_spring_names[torsional_spring]}_bifurcation")

        fig2 = plot.fig_create(2, 1, ("Damping", "Frequency"))
        for i in range(2*dofs):
            fig2.add_trace(
                go.Scatter(
                    x=vec_U, 
                    y=eigenvalues[i, :].real, 
                    mode="markers"
                ),
                row=1, 
                col=1
            )
            fig2.add_trace(
                go.Scatter(
                    x=vec_U, 
                    y=np.abs(eigenvalues[i, :].imag) * vec_U / b, 
                    mode="markers"
                ),
                row=2, 
                col=1
            )
        plot.fig_save(fig2, f"build/2dof/2dof_eigenvalues")
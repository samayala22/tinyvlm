import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import scipy as sp
from scipy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from tqdm import tqdm
from csaps import csaps
from numpy import genfromtxt

EPS_sqrt_f = np.sqrt(1.19209e-07)

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

def rk2_step(f: callable, x, t, dt, i):
    k1 = f(t, x[i])
    x_mid = x[i] + k1 * (dt / 2)
    t_mid = t + dt / 2
    k2 = f(t_mid, x_mid)
    x[i+1] = x[i] + 0.5 * (k1 + k2) * dt

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

def lee_monolithic_aeroelastic_system(ndv: NDVars):
    psi1 = 0.165
    psi2 = 0.335
    eps1 = 0.0455
    eps2 = 0.3

    c0 = 1 + 1 / ndv.mu
    c1 = ndv.x_a - ndv.a_h / ndv.mu
    c2 = (2 / ndv.mu) * (1 - psi1 - psi2)
    c3 = (1 / ndv.mu) * (1 + (1 - 2 * ndv.a_h) * (1 - psi1 - psi2))
    c4 = (2 / ndv.mu) * (eps1 * psi1 + eps2 * psi2)
    c5 = (2 / ndv.mu) * (1 - psi1 - psi2 + (0.5 - ndv.a_h) * (eps1 * psi1 + eps2 * psi2))
    c6 = (2 / ndv.mu) * eps1 * psi1 * (1 - eps1 * (0.5 - ndv.a_h))
    c7 = (2 / ndv.mu) * eps2 * psi2 * (1 - eps2 * (0.5 - ndv.a_h))
    c8 = -(2 / ndv.mu) * eps1**2 * psi1
    c9 = -(2 / ndv.mu) * eps2**2 * psi2

    r_alpha_sq = ndv.r_a ** 2
    d0 = (ndv.x_a / r_alpha_sq) - (ndv.a_h / (ndv.mu * r_alpha_sq))
    d1 = 1 + (1 + 8 * ndv.a_h**2) / (8 * ndv.mu * r_alpha_sq)
    d2 = -((1 + 2 * ndv.a_h) / (ndv.mu * r_alpha_sq)) * (1 - psi1 - psi2)
    d3 = ((1 - 2 * ndv.a_h) / (2 * ndv.mu * r_alpha_sq)) - \
         ((1 + 2 * ndv.a_h) * (1 - 2 * ndv.a_h) * (1 - psi1 - psi2)) / (2 * ndv.mu * r_alpha_sq)
    d4 = -((1 + 2 * ndv.a_h) / (ndv.mu * r_alpha_sq)) * (eps1 * psi1 + eps2 * psi2)
    d5 = -((1 + 2 * ndv.a_h) / (ndv.mu * r_alpha_sq)) * (1 - psi1 - psi2) - \
         ((1 + 2 * ndv.a_h) * (1 - 2 * ndv.a_h) * (psi1 * eps1 - psi2 * eps2)) / (2 * ndv.mu * r_alpha_sq)
    d6 = -((1 + 2 * ndv.a_h) * psi1 * eps1) / (ndv.mu * r_alpha_sq) * (1 - eps1 * (0.5 - ndv.a_h))
    d7 = -((1 + 2 * ndv.a_h) * psi2 * eps2) / (ndv.mu * r_alpha_sq) * (1 - eps2 * (0.5 - ndv.a_h))
    d8 = ((1 + 2 * ndv.a_h) * psi1 * eps1**2) / (ndv.mu * r_alpha_sq)
    d9 = ((1 + 2 * ndv.a_h) * psi2 * eps2**2) / (ndv.mu * r_alpha_sq)

    j = 1 / (c0 * d1 - c1 * d0)
    a21 = j * (-d5 * c0 + c5 * d0)
    a22 = j * (-d3 * c0 + c3 * d0)
    a23 = j * (-d4 * c0 + c4 * d0)
    a24 = j * (-d2 * c0 + c2 * d0)
    a25 = j * (-d6 * c0 + c6 * d0)
    a26 = j * (-d7 * c0 + c7 * d0)
    a27 = j * (-d8 * c0 + c8 * d0)
    a28 = j * (-d9 * c0 + c9 * d0)
    a41 = j * (d5 * c1 - c5 * d1)
    a42 = j * (d3 * c1 - c3 * d1)
    a43 = j * (d4 * c1 - c4 * d1)
    a44 = j * (d2 * c1 - c2 * d1)
    a45 = j * (d6 * c1 - c6 * d1)
    a46 = j * (d7 * c1 - c7 * d1)
    a47 = j * (d8 * c1 - c8 * d1)
    a48 = j * (d9 * c1 - c9 * d1)

    A = np.zeros((8, 8))
    A[0, 1] = 1
    A[1, 0] = a21 - j*c0*(1/ndv.U)**2
    A[1, 1] = a22 + 2*j*c0*ndv.zeta_a*(1/ndv.U)
    A[1, 2] = a23 + j*d0*(ndv.omega/ndv.U)**2
    A[1, 3] = a24 + 2*j*d0*ndv.zeta_h*(ndv.omega/ndv.U)
    A[1, 4] = a25
    A[1, 5] = a26
    A[1, 6] = a27
    A[1, 7] = a28
    A[2, 3] = 1
    A[3, 0] = a41 + j*c1*(1/ndv.U)**2
    A[3, 1] = a42 + 2*j*c1*ndv.zeta_a*(1/ndv.U)
    A[3, 2] = a43 - j*d1*(ndv.omega/ndv.U)**2
    A[3, 3] = a44 - 2*j*d1*ndv.zeta_h*(ndv.omega/ndv.U)
    A[3, 4] = a45
    A[3, 5] = a46
    A[3, 6] = a47
    A[3, 7] = a48
    A[4, 0] = 1
    A[4, 4] = -eps1
    A[5, 0] = 1
    A[5, 5] = -eps2
    A[6, 2] = 1
    A[6, 6] = -eps1
    A[7, 3] = 1
    A[7, 7] = -eps2

    return A

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

def fit_oscillatory_signal(time, signal):
    """
    Fits a damped or amplified oscillatory model to the provided signal.

    Parameters:
        time (np.ndarray): 1D array of time points.
        signal (np.ndarray): 1D array of signal values.

    Returns:
        popt (tuple): Optimal values for the parameters.
        pcov (2D array): Covariance of popt.
    """
    # Initial parameter guesses
    A_guess = (np.max(signal) - np.min(signal)) / 2
    C_guess = np.mean(signal)
    gamma_guess = -0.1 if np.all(np.diff(signal) < 0) else 0.1
    # Estimate frequency using FFT
    fft_vals = np.fft.fft(signal - C_guess)
    freqs = np.fft.fftfreq(len(time), d=(time[1] - time[0]))
    positive_freqs = freqs[freqs > 0]
    fft_magnitude = np.abs(fft_vals[freqs > 0])
    omega_guess = 2 * np.pi * positive_freqs[np.argmax(fft_magnitude)]
    phi_guess = 0

    initial_guesses = [A_guess, gamma_guess, omega_guess, phi_guess, C_guess]

    # Curve fitting
    popt, pcov = curve_fit(damped_oscillation, time, signal, p0=initial_guesses)
    return popt, pcov

def plot_fit(time, signal, fitted_signal):
    """
    Plots the original signal and the fitted model.

    Parameters:
        time (np.ndarray): Time vector.
        signal (np.ndarray): Original signal.
        fitted_signal (np.ndarray): Fitted signal from the model.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal, 'b.', label='Noisy Signal')
    plt.plot(time, fitted_signal, 'r-', label='Fitted Model')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.title('Oscillatory Signal Fitting')
    plt.show()


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
    def monolithic_system(t, y: np.ndarray):
        # Yung p212
        """
        system unknowns:
        q = [h, a, hd, ad, x1, x2]
        """
        M1 = np.zeros((6,6))
        M2 = np.zeros((6,6))
        V = np.zeros(6)

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
        f5 = (y0[1] + y0[2] + f0*y0[3])*(-psi1*np.exp(-eps1*t) - psi2*np.exp(-eps2*t))

        M1[0, 2] = 1
        M1[1, 3] = 1
        # M1[2, 0] = - (ndv.omega/ndv.U)**2
        M1[2, 1] = - 2 / ndv.mu
        M1[2, 2] = - 2.0*ndv.zeta_h*(ndv.omega/ndv.U) - 2 / ndv.mu
        M1[2, 3] = - (1/(np.pi*ndv.mu)) * (np.pi + 2*np.pi*(0.5 - ndv.a_h))
        M1[2, 4] = (2 / ndv.mu) * psi1
        M1[2, 5] = (2 / ndv.mu) * psi2
        # M1[3, 1] = - 1/(ndv.U**2) + f3*f4
        M1[3, 1] = f3*f4
        M1[3, 2] = f3*f4
        M1[3, 3] = - 2.0*ndv.zeta_a/ndv.U + f3*f4*f0 - f3*0.5*np.pi*f0
        M1[3, 4] = - psi1*f3*f4
        M1[3, 5] = - psi2*f3*f4
        M1[4, 3] = 1
        M1[4, 4] = -eps1
        M1[5, 3] = 1
        M1[5, 5] = -eps2

        V[2] = f1*f2*f5
        V[3] = f3*f4*f5
            
        V[2] += - (ndv.omega/ndv.U)**2 * y[0]
        V[3] += - 1/(ndv.U**2) * M(y[1])

        y_d = np.linalg.solve(M2, M1 @ y + V)
        # y_d = np.linalg.inv(M2) @ (M1 @ y + V)

        return y_d
    
    return monolithic_system

def fit_segment_weighted(freq_segment, psd_segment):
    # Define quadratic function
    def quad(x, a, b, c):
        return a*x**2 + b*x + c
    
    # quadratic polynomial distribution with close to 0 at each extremities
    def sigma_distribution(x): 
        a=-4/(freq_segment[0]-freq_segment[-1])**2
        b=-(2*a*freq_segment[0]-np.sqrt(-4*a))
        c=1+b**2/(4*a)
        return a*x**2+b*x+c+1e-2
    
    # sigma = np.ones_like(freq_segment)
    # sigma[0] = 1e-2 
    # sigma[-1] = 1e-2 
    sigma = sigma_distribution(freq_segment)

    p0 = np.polyfit(freq_segment, psd_segment, 2)
    
    # Fit with weighted points
    popt, _ = sp.optimize.curve_fit(
        quad,
        freq_segment,
        psd_segment,
        p0=p0, 
        sigma=sigma,
        absolute_sigma=True,
        maxfev=5000
    )
    
    return lambda x: quad(x, *popt)

def segment_and_fit_psd(frequencies, psd_db):
    # Find major peaks
    peaks, _ = sp.signal.find_peaks(psd_db, prominence=20)  # adjust prominence
    
    peaks_padded = np.zeros(len(peaks) + 2, dtype=peaks.dtype)
    peaks_padded[1:-1] = peaks
    peaks_padded[0] = 0
    peaks_padded[-1] = len(psd_db) - 1

    fitted_psd = np.zeros_like(psd_db)
    
    # Fit each segment between peaks
    for i in range(len(peaks_padded)-1):
        start = peaks_padded[i]
        end = peaks_padded[i+1]
        
        # Get segment data
        freq_segment = frequencies[start:end]
        psd_segment = psd_db[start:end]

        fit_func = fit_segment_weighted(freq_segment, psd_segment)
        fitted_psd[start:end] = fit_func(freq_segment)

    return fitted_psd

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
    # psd_db = segment_and_fit_psd(frequencies, psd_db_raw)

    return frequencies[mask], psd_db[mask]

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
    psi1 = 0.165
    psi2 = 0.335
    eps1 = 0.0455
    eps2 = 0.3

    # Dimensionless params
    flutter_speed = 6.285
    flutter_ratio = 0.2
    # vec_U = np.linspace(1.0, 7, 50)
    vec_U = [flutter_ratio * flutter_speed] # reduced velocity
    newton_err_thresh = 1e-7
    torsional_spring = 0
    torsional_spring_names = ["Freeplay", "Cubic", "Linear"]

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
        # t_final_nd = U_vel * 200.0
        t_final_nd = 100.0
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
        K = np.array([
            [(ndv.omega/ndv.U)**2, 0],
            [0, 1/(ndv.U**2)]
        ])
        zeros = np.zeros((2,2))

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
        a[:, 0] = np.linalg.solve(M, F[:, 0] + init_R - C @ v[:,0] - zeros @ u[:,0])
        def w(s: float):
            idx = int(s / dt_nd)
            return v[1, idx] + a[0, idx] + (0.5 - ndv.a_h) * a[1, idx]

        def dw1ds(s: float, w1: float): return w(s) - eps1 * w1
        def dw2ds(s: float, w2: float): return w(s) - eps2 * w2

        def aero(i):
            t = vec_t_nd[i]

            rk2_step(dw1ds, vec_w1, vec_t_nd[i-1], dt_nd, i-1)
            rk2_step(dw2ds, vec_w2, vec_t_nd[i-1], dt_nd, i-1)
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
                du, dv, da = newmark_beta_step(M, C, zeros, u[:,i], v[:,i], a[:,i], delta_F, dt_nd)

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
        
        # A = lee_monolithic_aeroelastic_system(ndv)
        # y0 = np.array([np.radians(3), 0, 0, 0, 0, 0, 0, 0])
        # monolithic_sol = solve_ivp(lambda t, y: A @ y, (0, t_final_nd), y0, t_eval=vec_t_nd)

        y0 = np.array([0, np.radians(3), 0, 0, 0, 0]) # h, a, hd, ad, x1, x2
        system = create_monolithic_system(y0, ndv, torsional_func)
        monolithic_sol = solve_ivp(system, (0, t_final_nd), y0, t_eval=vec_t_nd, method='RK45')

        # smoothed_h = sp.signal.savgol_filter(u[0, :], window_length=500, polyorder=3)
        # smoothed_h = sp.signal.savgol_filter(smoothed_h, window_length=800, polyorder=3)

        # sample_rate = 1 / (vec_t_nd[1] - vec_t_nd[0])  # Calculate sample rate from time array
        # smoothed_h = reconstruct_signal_minimizing_error(u[0, :], 10, sample_rate)

        peaks_h_t_i, peaks_h_d_i = get_peaks(vec_t_nd, u[0, :])
        peaks_a_t_i, peaks_a_d_i = get_peaks(vec_t_nd, u[1, :])

        if (len(vec_U) == 1):
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
            
            fig = make_subplots(
                rows=4, cols=3,
                subplot_titles=(
                    'Heave', 'Heave Velocity', 'Heave Force',
                    'Heave PSD', 'Heave Velocity PSD', 'Heave Force PSD',
                    'Pitch', 'Pitch Velocity', 'Pitch Force',
                    'Pitch PSD', 'Pitch Velocity PSD', 'Pitch Force PSD'
                ),
                vertical_spacing=0.1,
                horizontal_spacing=0.08
            )

            # Heave plots
            add_data_and_psd(fig, vec_t_nd, u[0, :], f"Iterative (ε = {newton_err_thresh})", 1, 1)
            add_data_and_psd(fig, monolithic_sol.t, monolithic_sol.y[0, :], "Monolithic", 1, 1)
            add_data_and_psd(fig, uvlm_t, uvlm_h, "UVLM", 1, 1, mode='markers')

            # Heave velocity plots
            add_data_and_psd(fig, vec_t_nd, v[0, :], f"Iterative (ε = {newton_err_thresh})", 1, 2)
            add_data_and_psd(fig, monolithic_sol.t, monolithic_sol.y[2, :], "Monolithic", 1, 2)
            add_data_and_psd(fig, uvlm_t, uvlm_hd, "UVLM", 1, 2, mode='markers')

            # Heave force plots
            add_data_and_psd(fig, vec_t_nd, F[0, :], f"Iterative (ε = {newton_err_thresh})", 1, 3)
            add_data_and_psd(fig, uvlm_t, uvlm_f_h, "UVLM", 1, 3, mode='markers')

            # Pitch plots (degrees)
            add_data_and_psd(fig, vec_t_nd, np.degrees(u[1, :]), f"Iterative (ε = {newton_err_thresh})", 3, 1)
            add_data_and_psd(fig, monolithic_sol.t, np.degrees(monolithic_sol.y[1, :]), "Monolithic", 3, 1)
            add_data_and_psd(fig, uvlm_t, np.degrees(uvlm_a), "UVLM", 3, 1, mode='markers')

            # Pitch velocity plots
            add_data_and_psd(fig, vec_t_nd, v[1, :], f"Iterative (ε = {newton_err_thresh})", 3, 2)
            add_data_and_psd(fig, monolithic_sol.t, monolithic_sol.y[3, :], "Monolithic", 3, 2)
            add_data_and_psd(fig, uvlm_t, uvlm_ad, "UVLM", 3, 2, mode='markers')

            # Pitch force plots
            add_data_and_psd(fig, vec_t_nd, F[1, :], f"Iterative (ε = {newton_err_thresh})", 3, 3)
            add_data_and_psd(fig, uvlm_t, uvlm_f_a, "UVLM", 3, 3, mode='markers')

            # Format all subplots
            # Time series plots - Row 1
            format_subplot(fig, 1, 1, r"$\bar{t}$", r"$\bar{h}$")
            format_subplot(fig, 1, 2, r"$\bar{t}$", r"$\bar{\dot{h}}$")
            format_subplot(fig, 1, 3, r"$\bar{t}$", r"$F_{h}$")
            
            # PSD plots - Row 2
            format_subplot(fig, 2, 1, r"$\bar{f}$", "Amplitude (dB)")
            format_subplot(fig, 2, 2, r"$\bar{f}$", "Amplitude (dB)")
            format_subplot(fig, 2, 3, r"$\bar{f}$", "Amplitude (dB)")
            
            # Time series plots - Row 3
            format_subplot(fig, 3, 1, r"$\bar{t}$", r"$\alpha$ (deg)")
            format_subplot(fig, 3, 2, r"$\bar{t}$", r"$\dot{\alpha}$")
            format_subplot(fig, 3, 3, r"$\bar{t}$", r"$F_{\alpha}$")
            
            # PSD plots - Row 4
            format_subplot(fig, 4, 1, r"$\bar{f}$", "Amplitude (dB)")
            format_subplot(fig, 4, 2, r"$\bar{f}$", "Amplitude (dB)")
            format_subplot(fig, 4, 3, r"$\bar{f}$", "Amplitude (dB)")

            fig.update_layout(
                title=f"2 DOF Aeroelastic response at Ū = {round(U_vel, 3)} ({torsional_spring_names[torsional_spring]} Pitch Spring)",
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

            fig.write_html("build/2dof.html", include_mathjax='cdn')


        # X = fft(u, axis=1)
        X = fft(monolithic_sol.y[0:2, :], axis=1)
        freq = fftfreq(n, d= b * dt_nd / U)
        max_freq_idx = np.argmax(np.abs(X), axis=1)
        dominant_freqs = np.abs(freq[max_freq_idx])
        freqs[:, idx] = 2*np.pi * dominant_freqs / omega_a

        damping_ratios[0, idx] = get_damping_ratio(u[0, :])
        damping_ratios[1, idx] = get_damping_ratio(u[1, :])
        damping_ratios_m[1, idx] = get_damping_ratio(monolithic_sol.y[1, :])

        def freq_ratio(time_range, nb_periods):
            m = ndv.mu * (np.pi * rho * b**2)
            omega_a = np.sqrt(k_a / (m * (ndv.r_a * b)**2))
            U = U_vel * (b * omega_a)
            total_t = b * time_range / U
            avg_period = total_t / nb_periods
            omega = 2*np.pi / avg_period
            return omega / omega_a

        # freqs[1, idx] = freq_ratio(peaks_a_t_i[-1] - peaks_a_t_i[0], len(peaks_a_t_i)-1)
        # freqs[0, idx] = freq_ratio(peaks_h_t_i[-1] - peaks_h_t_i[0], len(peaks_h_t_i)-1)

    if (len(vec_U) > 1):
        fig, axs = plt.subplot_mosaic(
            [["damping"], ["freq"]],  # Disposition des graphiques
            constrained_layout=True,  # Demander à Matplotlib d'essayer d'optimiser la disposition des graphiques pour que les axes ne se superposent pas
            figsize=(11, 8),  # Ajuster la taille de la figure (x,y)
        )
        # axs["damping"].plot(vec_U, damping_ratios[0, :], "o", label="Heave (Iterative)")
        # axs["damping"].plot(vec_U, damping_ratios[1, :], "o", label="Pitch (Iterative)")
        axs["damping"].plot(vec_U, damping_ratios_m[1, :], "o", label="Pitch (Monolithic)")

        xs = np.linspace(1, 7, 100)
        ys0 = csaps(vec_U, freqs[0, :], xs, smooth=0.993)
        ys1 = csaps(vec_U, freqs[1, :], xs, smooth=0.993)

        zhao_heave = genfromtxt('./data/zhao_heave.csv', delimiter=',')
        zhao_pitch = genfromtxt('./data/zhao_pitch.csv', delimiter=',')

        # axs["freq"].plot(vec_U, freqs[0, :], "o", label="Heave (Iterative)")
        # axs["freq"].plot(vec_U, freqs[1, :], "o", label="Pitch (Iterative)")
        axs["freq"].plot(xs, ys0, "o", markerfacecolor='none', mew=2, label="Heave")
        axs["freq"].plot(xs, ys1, "o", markerfacecolor='none', mew=2, label="Pitch")
        axs["freq"].plot(zhao_heave[:, 0], zhao_heave[:, 1], "D", markerfacecolor='none', mew=2, label="Zhao 2004")
        axs["freq"].plot(zhao_pitch[:, 0], zhao_pitch[:, 1], "D", markerfacecolor='none', mew=2, label="Zhao 2004")
        axs["freq"].axvline(x=flutter_speed, color='r', linestyle='--', label='Flutter Speed')

        axs["damping"].grid(True, which='both', linestyle=':', linewidth=1.0, color='gray')
        axs["damping"].legend()
        axs["damping"].set_xlabel('U')
        axs["damping"].set_ylabel(r"$\zeta$")

        axs["freq"].grid(True, which='both', linestyle=':', linewidth=1.0, color='gray')
        axs["freq"].legend()
        axs["freq"].set_xlabel(r'Reduced Velocity $\bar{U}$')
        axs["freq"].set_ylabel(r"Frequency ratio $\omega / \omega_{\alpha}$")

        plt.show()

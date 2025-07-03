import numpy as np

import finite_diff as fd

class Dims():
    def __init__(self, n_d, n_h):
        # assert ((n_h+1) & n_h) == 0 # H+1 should be a power of 2
        self.n_d = n_d
        self.n_h = n_h
        self.n_s = (n_h + 1) * (2**5)
        self.n_c = 2 * n_h + 1
        self.n_u = self.n_d * self.n_c

def create_lanczos_filter(N, m=1):
    L = np.zeros(N)
    L[0] = 1
    def sinc(x): return np.sin(np.pi * x) / (np.pi*x)
    for i in range(1, N+1):
        xi = i / (N+1)
        L[i-1] = sinc(xi) ** m
    return L

def X_to_complex(X):
    """
    Converts a dofs * (2H+1) real array [A0, A1, B1, ... A_H, B_H] to a dofs * (H+1) comlex array
    """
    assert len(X.shape) == 2 # matrix form
    dofs = X.shape[0]
    H = int((X.shape[1] - 1) / 2)
    Xc = np.zeros((dofs, H+1), dtype=np.complex128)
    for d in range(dofs):
        Xc[d, 0] = X[d, 0] - 0j
    for h in range(1, H+1):
        for d in range(dofs):
            Xc[d, h] = (X[d, 2*h-1] - 1j * X[d, 2*h])/2
    return Xc

def X_to_real(X, lanczos_m=1.0):
    """
    Converts a dofs * N complex array to a dofs * (2*N-1) real array
    """
    assert len(X.shape) == 2 # matrix form
    dofs = X.shape[0]
    N = X.shape[1]
    L = create_lanczos_filter(N, lanczos_m)
    Xr = np.zeros((dofs, 2*N-1), dtype=np.float64)
    for d in range(dofs):
        Xr[d, 0] = X[d, 0].real
    for h in range(1, N):
        for d in range(dofs):
            Xr[d, 2*h-1] = L[h] * 2 * X[d, h].real
            Xr[d, 2*h] = L[h] * -2 * X[d, h].imag
    
    return Xr

def nabla(H):
    nabla = np.zeros((2*H+1, 2*H+1))
    nabla_j = np.array([[0, 1], [-1, 0]])
    for j in range(1, H+1):
        nabla[2*j-1:2*j+1, 2*j-1:2*j+1] = j*nabla_j
    return nabla

def linear_residual(X, motion, dims):
    omega_idx = dims.n_u - X.shape[0]
    Om = X[omega_idx]
    param = X[-1]
    sys = motion()
    M = sys.M(param)
    C = sys.C(param)
    K = sys.K(param)

    R_lin = np.zeros(dims.n_u)
    nab = nabla(dims.n_h)
    Z = np.kron(Om**2 * nab @ nab, M) + np.kron(Om * nab, C) + np.kron(np.eye(dims.n_c), K)
    R_lin = Z @ X[:omega_idx]
    return R_lin

def nonlinear_residual(X, motion, dims):
    omega_idx = dims.n_u - X.shape[0]
    lanczos_m = 0.0
    Om = X[omega_idx]
    param = X[-1]
    sys = motion()

    Xc_real = X[:omega_idx].reshape(dims.n_c, dims.n_d).T
    Xc = X_to_complex(Xc_real)
    k = np.arange(dims.n_h + 1)
    q = np.fft.irfft(Xc, dims.n_s, axis=1, norm='forward')
    q_dot = np.fft.irfft(1j * Om * k * Xc, dims.n_s, axis=1, norm='forward')
    R_nlt = np.zeros((dims.n_d, dims.n_s))
    
    for s in range(dims.n_s):
        R_nlt[:, s] = - sys.fnlt(Xc_real, q[:, s], q_dot[:, s], Om, param)
    
    R_nl_fft = np.fft.rfft(R_nlt, dims.n_s, axis=1, norm='backward')
    R_nl = X_to_real(R_nl_fft[:, :dims.n_h+1] / dims.n_s, lanczos_m).T.reshape(-1)
    
    R_nlft = - sys.fnlf(Xc_real, Om, param)
    R_nlf_fft = np.fft.rfft(R_nlft, dims.n_c, axis=1, norm='backward')
    R_nlf = X_to_real(R_nlf_fft / dims.n_c, lanczos_m).T.reshape(-1)

    return R_nl + R_nlf

def residual(X, *args):
    return linear_residual(X, *args) + nonlinear_residual(X, *args)

def jacobian(X, motion, dims):
    omega_idx = dims.n_u - X.shape[0]
    Om = X[omega_idx]
    param = X[-1]
    sys = motion()
    M = sys.M(param)
    C = sys.C(param)
    K = sys.K(param)
    dMdU = sys.dMdU(param)
    dCdU = sys.dCdU(param)
    dKdU = sys.dKdU(param)

    nab = nabla(dims.n_h)

    J_lin = np.zeros((dims.n_u, X.shape[0]))

    J_lin[:, :omega_idx] = np.kron(Om**2 * nab @ nab, M) + np.kron(Om * nab, C) + np.kron(np.eye(dims.n_c), K)
    J_lin[:, omega_idx] = (np.kron(2*Om * nab @ nab, M) + np.kron(nab, C)) @ X[:omega_idx]
    if omega_idx == -2:
        J_lin[:, -1] = (np.kron(Om**2 * nab @ nab, dMdU) + np.kron(Om * nab, dCdU) + np.kron(np.eye(dims.n_c), dKdU)) @ X[:omega_idx]
    
    # J_lin2 = fd.numerical_jac(linear_residual, X, motion, dims)
    # np.testing.assert_allclose(J_lin, J_lin2)

    J_nlin = fd.numerical_jac(nonlinear_residual, X, motion, dims)
    return J_lin + J_nlin

def integral_orthogonal_phase_condition(X, X_ref, motion, dims):
    X_mat = X[:-2].reshape(dims.n_c, dims.n_d).T
    X_mat_ref = X_ref[:-2].reshape(dims.n_c, dims.n_d).T
    orthogonality = 0
    for k in range(1, dims.n_h+1):
        orthogonality += k * (np.dot(X_mat_ref[:, 2*k], X_mat[:, 2*k-1]) - np.dot(X_mat_ref[:, 2*k-1], X_mat[:, 2*k]))
    return orthogonality

def dintegral_orthogonal_phase_condition(X, X_ref, motion, dims):
    X_mat_ref = X_ref[:-2].reshape(dims.n_c, dims.n_d).T
    jac_row = np.zeros(X.shape[0])
    for k in range(1, dims.n_h + 1):
        col_2k_minus_1 = int((2*k - 1) * dims.n_d)
        col_2k = int(2*k * dims.n_d)
        jac_row[col_2k_minus_1:col_2k_minus_1+dims.n_d] = k * X_mat_ref[:, 2*k]
        jac_row[col_2k:col_2k + dims.n_d] = - k * X_mat_ref[:, 2*k - 1]
    jac_row[-2:] = 0.0
    return jac_row

def truncated_series_approximation(dt, u_tr, dims):
    N_tr = u_tr.shape[1]  # Number of time samples
    dc = np.mean(u_tr, axis=1)
    u_tr = u_tr - dc[:, None] # offset mean value to prevent spectral leakage
    window = np.hanning(N_tr)
    u_tr_windowed = u_tr * window[None, :]  # Multiply each DoF by the window
    zp_factor = 4                   # zero padding factor
    N_fft = zp_factor * N_tr        # New FFT length after padding
    U_fft = np.fft.fft(u_tr_windowed, n=N_fft, axis=1)
    norm_factor = window.sum()
    freqs = np.fft.fftfreq(N_fft, dt)
    pos = freqs > 0
    f_pos = freqs[pos]
    U_pos = U_fft[:, pos]
    ref_dof = 0
    amplitude_ref = np.abs(U_pos[ref_dof, :])
    i0_ref = np.argmax(amplitude_ref)
    f0 = f_pos[i0_ref]
    omega0 = 2 * np.pi * f0        # Base angular frequency
    print("Base frequency: {:.3f} rad/s".format(omega0))

    coeffs = np.zeros((dims.n_d, dims.n_c))
    coeffs[:, 0] = dc
    for h in range(1, dims.n_h + 1):
        target = h * f0
        idx = np.argmin(np.abs(f_pos - target))  # Find the closest frequency bin
        Y = U_pos[:, idx]
        # Multiply by 2 because of the use of a one-sided FFT (except the DC term)
        coeffs[:, 2 * h - 1] = 2 * np.real(Y) / norm_factor   # cosine coefficient
        coeffs[:, 2 * h]     = -2 * np.imag(Y) / norm_factor   # sine coefficient

    return coeffs, omega0

def create_fourier_basis(omega, harmonics, t):
    unknowns = 2 * harmonics + 1
    basis = np.zeros((unknowns))
    dbasis = np.zeros((unknowns))
    ddbasis = np.zeros((unknowns))
    basis[0] = 1
    dbasis[0] = 0
    ddbasis[0] = 0
    for i in range(harmonics):
        k = float(i+1)
        basis[2 * i + 1] = np.cos(omega * t * k)
        basis[2 * i + 2] = np.sin(omega * t * k)
        dbasis[2 * i + 1] = - omega * k * np.sin(omega * t * k)
        dbasis[2 * i + 2] = omega * k * np.cos(omega * t * k)
        ddbasis[2 * i + 1] = - (omega * k)**2 * np.cos(omega * t * k)
        ddbasis[2 * i + 2] = - (omega * k)**2 * np.sin(omega * t * k)

    return basis, dbasis, ddbasis

def to_timedomain(t_begin, t_end, dt, dofs, X, omega, harmonics):
    # Plot the result in time domain
    samples = 2*harmonics+1
    vec_t = np.arange(t_begin, t_end + dt, dt)
    sol = np.zeros((3*dofs, vec_t.shape[0])) # u, v, a
    uf_sol_ = X.reshape(samples, dofs).T
    # uf_sol_ = xf0.reshape(samples, dofs).T
    for i, t in enumerate(vec_t):
        b, db, ddb = create_fourier_basis(omega, harmonics, t)
        sol[0:dofs, i] = uf_sol_ @ b
        sol[dofs:2*dofs, i] = uf_sol_ @ db
        sol[2*dofs:3*dofs, i] = uf_sol_ @ ddb

    return vec_t, sol


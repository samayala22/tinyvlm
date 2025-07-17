import argparse, hashlib, pickle, json, time
from enum import Enum

import numpy as np
import scipy as sp
import plotly.graph_objects as go

import harmonic_balance as hb # temporary
import finite_diff as fd
import plotting as plot
import helpers

BIFURCATIONS = ["LP", "BP", "NS", "PD"] # Maybe change to enum ?

class Parametrisation(Enum):
    Local = 1
    ArcLength = 2

class Metadata:
    def __init__(self):
        self.name = ""
        self.param_start = 0.0
        self.param_end = 0.0
        self.max_steps = 5000
        self.scaling = False
        self.step_adapt = False
        self.ds = [] # Continuation step size
        self.bifurcation_test = None # History of test functions
        self.bifurcation = {"LP": [], "BP": [], "NS": [], "PD": []} # Bifurcation indices position
        self.stable = []
        self.floquet_exponents = None
        self.X = None
        self.dims = None

def bialternate(a, b):
    """
    https://webspace.science.uu.nl/~kouzn101/NBA/Bialt.pdf
    """
    assert a.ndim == 2
    assert b.ndim == 2
    if a.shape[0] != a.shape[1] or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("Input matrices must be square and of the same shape.")
    n = a.shape[0]
    assert n > 1
    
    # m = n * (n - 1) // 2
    # c = np.zeros((m, m))
    # i = 0
    # for p in range(1, n):
    #     for q in range(p):
    #         j = 0
    #         for r in range(1, n):
    #             for s in range(r):
    #                 val = 0.5 * (
    #                     a[p, r] * b[q, s] - a[p, s] * b[q, r] +
    #                     a[q, s] * b[p, r] - a[q, r] * b[p, s]
    #                 )
    #                 c[i, j] = val
    #                 j += 1
    #         i += 1

    idx_pairs = np.array([(q, p) for p in range(1, n) for q in range(p)])
    q_i, p_i = idx_pairs.T
    s_j, r_j = idx_pairs.T
    p_i, q_i = p_i[:, np.newaxis], q_i[:, np.newaxis]
    r_j, s_j = r_j[np.newaxis, :], s_j[np.newaxis, :]
    term1 = a[p_i, r_j] * b[q_i, s_j]
    term2 = a[p_i, s_j] * b[q_i, r_j]
    term3 = a[q_i, s_j] * b[p_i, r_j]
    term4 = a[q_i, r_j] * b[p_i, s_j]
    c = 0.5 * (term1 - term2 + term3 - term4)
    return c

def lp_test(om_i, om_im1):
    """
    Limit point test function (Delbé 3.3)
    """
    diff = om_i - om_im1
    return diff / np.abs(diff)

def bp_test(J):
    """
    Branch point test function (Dimitriadis 7.43)
    """
    return np.linalg.det(J)

def ns_test(floquet_exponents):
    """
    Neimark-Sacker test function (Colaitis 7.3.9)
    """
    A_tilde = np.diag(floquet_exponents)
    res = np.linalg.det(2.0 * bialternate(A_tilde, np.eye(A_tilde.shape[0])))
    return res.real

def pd_test(floquet_exponents, omega):
    """
    Period doubling test function (Colaitis 7.3.13)
    """
    T = 2 * np.pi / omega
    floquet_multipliers = np.exp(floquet_exponents * T)
    max_multiplier = np.max(np.abs(floquet_multipliers))
    return np.sign(np.real(max_multiplier)) * np.abs(max_multiplier) + 1

def tangent_predictor(J, zref, Xref):
    """Compute tangent vector using Seydel's pivot strategy."""
    # 1. Determine pivot indices
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_changes = np.abs(zref) / np.maximum(np.abs(Xref), 1e-4)
    kk = np.argsort(-rel_changes)  # Descending order
    
    # 2. Try different pivots until success
    ztmp = None
    for k in kk:
        # 3. Create constraint vector
        c = np.zeros_like(Xref)
        c[k] = 1.0
        
        # 4. Build extended system
        J_red = J[:-1, :]  # Exclude last row (parameter derivative)
        A = np.vstack([J_red, c])
        b = np.concatenate([np.zeros(J_red.shape[0]), [1.0]])
        
        # 5. Solve with least-squares for numerical stability
        ztmp, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        if not np.any(np.isnan(ztmp)):
            break
    
    # 6. Normalize tangent vector
    # z = ztmp / np.linalg.norm(ztmp) # length 1 vector
    return ztmp

# def tangent_predictor(J, zref, Xref):
#     Q, R = np.linalg.qr(J.T)
#     z = Q[:, -1]
#     return z / np.linalg.norm(z)

def extended_residual(
    X, # scaled
    X_ref,
    z_ref,
    Dscale,
    parametrisation: Parametrisation,
    ds,
    omega_idx,
    *args
):
    X_unscaled = X * Dscale # unscaled X
    ext_res = np.zeros_like(X)
    ext_res[:omega_idx] = hb.residual(X_unscaled, *args) # TODO: take as input
    if omega_idx == -2:
        ext_res[-2] = hb.integral_orthogonal_phase_condition(X_unscaled, X_ref * Dscale, *args)

    if parametrisation == Parametrisation.Local:
        ext_res[-1] = np.dot(z_ref, X - X_ref)
    elif parametrisation == Parametrisation.ArcLength:
        ext_res[-1] = np.dot(X - X_ref, X - X_ref) - ds**2 # iteration on a normal plane, perpendicular to tangent

    return ext_res

def extended_residual_jacobian(X, X_ref, z_ref, Dscale, parametrisation, ds, omega_idx, *args):
    Jext = np.zeros((X.shape[0], X.shape[0]))
    # Jext[:omega_idx, :] = fd.numerical_jac(hb.residual, X * Dscale, *args)
    Jext[:omega_idx, :] = hb.jacobian(X * Dscale, *args)
    
    if omega_idx == -2:
        Jext[-2, :] = hb.dintegral_orthogonal_phase_condition(X * Dscale, X_ref * Dscale, *args)

    # Parametrisation
    if parametrisation == Parametrisation.Local:
        Jext[-1, :] = z_ref
    elif parametrisation == Parametrisation.ArcLength:
        Jext[-1, :] = 2 * (X - X_ref)

    # Scaling
    Jext[:-1, :] = Jext[:-1, :] @ np.diag(Dscale)
    
    return Jext

def solve_nonlinear_system(X0, X_ref, z_ref, Dscale, parametrisation, ds, omega_idx, *args):
    st = time.perf_counter_ns()
    sol = sp.optimize.root(
        extended_residual,
        X0,
        args=(X_ref, z_ref, Dscale, parametrisation, ds, omega_idx, *args),
        method='hybr',
        jac=extended_residual_jacobian,
        tol = 1e-6,
    )

    if not sol.success:
        print(f"Nonlinear solver failed: {sol.message}")
        return False, ()

    # Jacobian assembled from QR decomposition
    Q = sol.fjac.T
    R = np.zeros_like(sol.fjac)
    R[np.triu_indices_from(R)] = sol.r
    jac = Q @ R

    et = time.perf_counter_ns()

    return True, (sol.x, jac, sol['nfev'], sol['njev'], (et-st)*1e-9)

def hash_metadata(metadata):
    params = {
        "name":         metadata.name,
        "start":  metadata.X[-1, 0],
        "end":   metadata.X[-1, -1],
        "iterations": metadata.X.shape[1]
    }
    j = json.dumps(params, sort_keys=True).encode("utf8")
    return hashlib.md5(j).hexdigest()[:8]

def stability_analysis(J_ext, motion, omega, omega_idx, dims):
    """
    Hill stability analysis using the Quadratic Eigenvalue Problem formulation.
    """
    J_hb = J_ext[:omega_idx, :omega_idx]  # Exclude the last two rows/columns (parameter and orthogonality condition)
    hb_dim = dims.n_d * dims.n_c # dofs * (2 * H + 1)
    H = np.zeros((2 * hb_dim, 2 * hb_dim)) # first order casted system
    nab = hb.nabla(dims.n_h)
    system = motion()
    M = system.M(omega)
    C = system.C(omega)
    Lambda1 = np.kron(2.0 * omega * nab, M) + np.kron(np.eye(dims.n_c), C) 
    Lambda2 = np.kron(np.eye(dims.n_c), M)

    H[0:hb_dim, hb_dim:2*hb_dim] = np.eye(hb_dim)
    H[hb_dim:2*hb_dim, 0:hb_dim] = -np.linalg.inv(Lambda2) @ J_hb
    H[hb_dim:2*hb_dim, hb_dim:2*hb_dim] = -np.linalg.inv(Lambda2) @ Lambda1

    eigvals = np.linalg.eigvals(H)
    floquet_exponents = eigvals[np.argsort(np.abs(np.imag(eigvals)))[:2*dims.n_d]]
    max_real_part = np.max(np.real(floquet_exponents))
    is_stable = max_real_part < 1e-8
    return is_stable, floquet_exponents

def continuation(X0, motion, metadata: Metadata):
    """
    Continuation for autonomous systems using the harmonic balance method
    """
    # print("Initial guess X0:", X0)
    ds = metadata.ds[0]
    ds_min = ds / 5.0
    ds_max = ds * 5.0
    n_opt = 6 + X0.shape[0] * 1 # optimal nubmer of function evals
    omega_idx = metadata.dims.n_u - X0.shape[0]
    
    X_mat = np.zeros((X0.shape[0], metadata.max_steps))
    metadata.floquet_exponents = np.zeros((2 * metadata.dims.n_d, metadata.max_steps), dtype=np.complex128)
    metadata.bifurcation_test = np.zeros((4, metadata.max_steps))
    metadata.stable = np.zeros(metadata.max_steps, dtype=np.bool)
    metadata.ds = np.zeros(metadata.max_steps)
    
    if metadata.param_end > metadata.param_start:
        param_direction = 1
        direction = 1
    else:
        param_direction = -1
        direction = -1

    J = np.zeros((X0.shape[0], X0.shape[0]))  # Jacobian
    X_ref = X0.copy()
    X_old = X0.copy()
    z_ref = np.zeros_like(X0)
    z_ref[-1] = 1

    Dscale = np.ones_like(X0)
    Dscale_prev = Dscale.copy()

    iteration = 0
    try:
        while iteration < metadata.max_steps:
            if metadata.scaling:
                Dscale_prev = Dscale.copy()
                Dscale = np.maximum(np.abs(X0 * Dscale_prev), np.ones_like(X0))
                Dscale[omega_idx] = 1.0 # omega is not scaled
                Dscale[-1] = 1.0 # param is not scaled
                X_ref = X_ref * (Dscale_prev / Dscale)
                X0 = X0 * (Dscale_prev / Dscale)
                X_old = X_old * (Dscale_prev / Dscale)

            if iteration == 0:
                parametrisation = Parametrisation.Local
                Xp = X0.copy()
            else:
                parametrisation = Parametrisation.ArcLength
                # J = extended_residual_jacobian(X0, X_ref, z_ref * Dscale_prev, hb_residual, Dscale, Parametrisation.ArcLength, True, False)
                ztmp = tangent_predictor(J @ np.diag(1 / Dscale_prev), z_ref * Dscale_prev, X_ref) / Dscale
                z = ztmp / np.linalg.norm(ztmp)

                # Take a step in the tangent direction ensuring to stay along the solution path
                if (iteration > 1) and np.dot(X0-X_old, direction*ds*z) < 0:
                    direction *= -1

                X_ref = X0.copy()
                z_ref = z.copy()
                # Predictor step
                Xp = X0 + direction*ds*z

            # Corrector step
            success, results = solve_nonlinear_system(Xp, X_ref, z_ref, Dscale, parametrisation, ds, omega_idx, motion, metadata.dims)
            if not success:
                if ds > ds_min and iteration > 0:
                    print(f"Nonlinear solver failed, reducing step size from {ds:.3f} to {ds/2:.3f}")
                    ds /= 2.0
                    continue
                else:
                    print("Nonlinear solver failed, continuation stopped")
                    break

            Xtmp, J, nfev, njev, timing = results
            
            # Bookkeeping
            X_old = X0.copy()
            X0 = Xtmp.copy()
            X = X0 * Dscale # unscaled X
            X_mat[:, iteration] = X

            # Stability analysis
            is_stable, floquet_exponents = stability_analysis(J, motion, X[omega_idx], omega_idx, metadata.dims)
            metadata.floquet_exponents[:, iteration] = floquet_exponents
            metadata.stable[iteration] = is_stable
            metadata.ds[iteration] = ds

            # Bifurcation tests
            metadata.bifurcation_test[0, iteration] = lp_test(X[-1], X_mat[-1, iteration-1]) if iteration > 0 else 0.0
            metadata.bifurcation_test[1, iteration] = bp_test(J)
            metadata.bifurcation_test[2, iteration] = ns_test(floquet_exponents)
            metadata.bifurcation_test[3, iteration] = pd_test(floquet_exponents, X[omega_idx])

            # Bifurcation detection
            if iteration > 1: # first iteration is inaccurate
                for i, name in enumerate(BIFURCATIONS):
                    if np.copysign(1.0, metadata.bifurcation_test[i, iteration-1]) != np.copysign(1.0, metadata.bifurcation_test[i, iteration]):
                        print(f"{name} bifurcation detected")
                        metadata.bifurcation[name].append(iteration)
            
            print(f"{iteration} | ds: {ds:.4f}, param: {X[-1]:.3f}, omega: {X[omega_idx]:.3f}, stable: {is_stable}, nfev: {nfev}, njev: {njev}, timing: {timing:.2f}s")

            if metadata.step_adapt:
                n_f = nfev + J.shape[1] * njev
                xi = n_opt / n_f
                xi = np.clip(xi, 0.5, 2.0)
                ds = np.clip(ds * xi, ds_min, ds_max)

            iteration += 1
            
            if (X[-1] - metadata.param_end) * param_direction >= 0:
                print("Continuation reached the end")
                break
    except KeyboardInterrupt:
        print("Continuation interrupted by user")

    metadata.X = X_mat[:, :iteration]
    metadata.stable = metadata.stable[:iteration]
    metadata.floquet_exponents = metadata.floquet_exponents[:, :iteration]
    metadata.bifurcation_test = metadata.bifurcation_test[:, :iteration]
    metadata.ds = metadata.ds[:iteration]

    filename = f"continuation_{hash_metadata(metadata)}.pkl"
    print(f"Continuation data saved to {filename}")
    with open(f"build/{filename}", 'wb') as f:
        pickle.dump(metadata, f)

    return metadata

def plot_hb_continuation(metadata):
    dofs = metadata.dims.n_d
    omega_idx = metadata.dims.n_u - metadata.X.shape[0]
    print(f"dofs: {dofs}")
    
    fig = plot.fig_create(dofs+1, 1, tuple(f"DOF {i+1}" for i in range(dofs)), "Continuation")
    
    stable_mask = np.array(metadata.stable)
    stable_mask2 = stable_mask.copy()
    X_stable = metadata.X.copy()
    X_unstable = metadata.X.copy()
    
    # Padding to make the line continuous between stable and unstable regions
    for i in range(len(stable_mask)-1):
        if stable_mask[i+1] == False and stable_mask[i] == True:
            stable_mask2[i+1] = True
        elif stable_mask[i+1] == True and stable_mask[i] == False:
            stable_mask2[i] = True

    X_stable[:, ~stable_mask2] = np.nan
    X_unstable[:, stable_mask] = np.nan
    
    for dof in range(dofs):
        X_h = metadata.X[dof:omega_idx:dofs, :]
        X_h_stable = X_stable[dof:omega_idx:dofs, :]
        X_h_unstable = X_unstable[dof:omega_idx:dofs, :]

        for h in range(1, metadata.dims.n_h+1):
            name = f"Harmonic {h}"
            fig.add_trace(
                go.Scattergl(
                    x = X_stable[-1, :],
                    y = np.sqrt(X_h_stable[2*h-1, :]**2 + X_h_stable[2*h, :]**2),
                    name = name,
                    legendgroup = name,
                    mode = "lines",
                    line = {"dash": "solid"},
                    showlegend = True if dof == 0 else False
                ),
                row=dof+1,
                col=1
            )
            fig.add_trace(
                go.Scattergl(
                    x = X_unstable[-1, :],
                    y = np.sqrt(X_h_unstable[2*h-1, :]**2 + X_h_unstable[2*h, :]**2),
                    name = name,
                    legendgroup = name,
                    mode = "lines",
                    line = {"dash": "dash"},
                    showlegend = False
                ),
                row=dof+1,
                col=1
            )

        symbols = ["circle", "square", "diamond", "triangle"]
        for i, b in enumerate(BIFURCATIONS):
            h=1 # only plot the bifurcations for the first harmonic
            indices = np.array(metadata.bifurcation[b])
            if len(indices) == 0: continue
            fig.add_trace(
                go.Scattergl(
                    x = metadata.X[-1, indices],
                    y = np.sqrt(X_h[2*h-1, indices]**2 + X_h[2*h, indices]**2),
                    name = b,
                    legendgroup = b,
                    mode = "markers",
                    marker = {"size": 10, "symbol": symbols[i]},
                    showlegend = True if dof == 0 else False
                ),
                row=dof+1,
                col=1
            )

        plot.format_subplot(fig, dof+1, 1, r"$\bar{U}$", f"$||H_{dof+1}||^{2}$")

    plot.fig_save(fig, f"build/continuation/continuation_{hash_metadata(metadata)}", pdf=False)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    with open(args.filename, 'rb') as f:
        metadata = pickle.load(f)
    print(f"Hash: {hash_metadata(metadata)}")
    plot_hb_continuation(metadata)


# REFERENCES:
# Dimitriadis: INTRODUCTION TO NONLINEAR AEROELASTICITY 
# Colaitis: Stratégie numérique pour l'analyse qualitative des interactions aube/carter
# Delbé: Application d'une stratégie numérique de suivi de bifurcations à l'analyse d'interactions aube/carter dans les moteurs d'avions

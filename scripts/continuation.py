import argparse, hashlib, pickle, json
from enum import Enum

import numpy as np
import scipy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import harmonic_balance as hb # temporary
import finite_diff as fd
import plotting as plot
import helpers

EPS = np.finfo(np.float64).eps
def cd2_h(x0): return np.maximum(np.where(np.abs(x0) > 1, np.cbrt(EPS * np.abs(x0)), np.cbrt(EPS) * np.abs(x0)), EPS)
def fd_h(x0): return np.maximum(np.sqrt(EPS) * np.abs(x0), EPS)

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
        self.ds = []
        self.fold_pt = []
        self.branch_pt = []
        self.X = None
        self.dims = None

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
    *args
):
    X_unscaled = X * Dscale # unscaled X
    ext_res = np.zeros_like(X)
    ext_res[:-2] = hb.residual(X_unscaled, *args) # TODO: take as input
    ext_res[-2] = hb.integral_orthogonal_phase_condition(X_unscaled, X_ref * Dscale, *args) # TODO: take as input

    if parametrisation == Parametrisation.Local:
        ext_res[-1] = np.dot(z_ref, X - X_ref)
    elif parametrisation == Parametrisation.ArcLength:
        ext_res[-1] = np.dot(X - X_ref, X - X_ref) - ds**2 # iteration on a normal plane, perpendicular to tangent

    return ext_res

def extended_residual_jacobian_hybrid(X, X_ref, z_ref, Dscale, parametrisation, ds, *args):
    Jext = np.zeros((X.shape[0], X.shape[0]))
    Jext[:-2, :] = hb.jacobian(X * Dscale, *args)
    
    Jext[-2, :] = hb.dintegral_orthogonal_phase_condition(X * Dscale, X_ref * Dscale, *args) # TODO: take as input

    # Parametrisation
    if parametrisation == Parametrisation.Local:
        Jext[-1, :] = z_ref
    elif parametrisation == Parametrisation.ArcLength:
        Jext[-1, :] = 2 * (X - X_ref)

    # Scaling
    Jext[:-1, :] = Jext[:-1, :] @ np.diag(Dscale)
    
    return Jext

def extended_residual_jacobian(X, X_ref, z_ref, Dscale, parametrisation, ds, *args):
    Jext = np.zeros((X.shape[0], X.shape[0]))
    Jext[:-2, :] = fd.numerical_jac(hb.residual, X * Dscale, *args)
    Jext[-2, :] = hb.dintegral_orthogonal_phase_condition(X * Dscale, X_ref * Dscale, *args) # TODO: take as input

    # Parametrisation
    if parametrisation == Parametrisation.Local:
        Jext[-1, :] = z_ref
    elif parametrisation == Parametrisation.ArcLength:
        Jext[-1, :] = 2 * (X - X_ref)

    # Scaling
    Jext[:-1, :] = Jext[:-1, :] @ np.diag(Dscale)
    
    return Jext

@helpers.Timing("Solver:")
def solve_nonlinear_system(X0, X_ref, z_ref, Dscale, parametrisation, ds, *args):
    sol = sp.optimize.root(
        extended_residual,
        X0,
        args=(X_ref, z_ref, Dscale, parametrisation, ds, *args),
        method='hybr',
        jac=extended_residual_jacobian,
        tol = 1e-6,
    )

    if not sol.success:
        print(f"Nonlinear solver failed: {sol.message}")
        return None, None, None, False

    Q = sol.fjac.T
    R = np.zeros_like(sol.fjac)
    R[np.triu_indices_from(R)] = sol.r
    jac = Q @ R

    print(f"param: {sol.x[-1]:.3f}, omega: {sol.x[-2]:.3f}, nfev: {sol.get('nfev', None)}, njev: {sol.get('njev', None)}")

    nfev = sol['nfev'] + 2*jac.shape[1]*sol['njev']
    return sol.x, jac, nfev, True

def hash_metadata(metadata):
    params = {
        "name":         metadata.name,
        "start":  metadata.X[-1, 0],
        "end":   metadata.X[-1, -1],
        "iterations": metadata.X.shape[1]
    }
    j = json.dumps(params, sort_keys=True).encode("utf8")
    return hashlib.md5(j).hexdigest()[:8]

def plot_hb(X, iteration, dims):
    fig = plot.create_dofs_figure(["Heave", "Pitch"], f"U = {X[-1]:.2f}")
    hb_sol_t, hb_sol = hb.to_timedomain(0.0, 1000.0, 0.1, dims.n_d, X[:-2], X[-2], dims.n_h)
    plot.add_data_and_psd(fig, hb_sol_t, hb_sol[0, :], "HB-VLM", 1, 1, 1)
    plot.add_data_and_psd(fig, hb_sol_t, hb_sol[1, :], "HB-VLM", 3, 1, 1)
    plot.add_data_and_psd(fig, hb_sol_t, hb_sol[2, :], "HB-VLM", 1, 2, 1)
    plot.add_data_and_psd(fig, hb_sol_t, hb_sol[3, :], "HB-VLM", 3, 2, 1)
    # param_str = f"{X[-1]:.2f}".replace('.', '_')
    plot.fig_save(fig, f"build/continuation/cont_{iteration}")

def continuation(X0, motion, metadata):
    """
    Continuation for autonomous systems using the harmonic balance method
    """
    # print("Initial guess X0:", X0)
    ds = metadata.ds[0]
    ds_min = ds / 5.0
    ds_max = ds * 5.0
    nfev_opt = 5 + 2 * X0.shape[0] * 1 # optimal nubmer of function evals
    
    X_mat = np.zeros((X0.shape[0], metadata.max_steps))

    if metadata.param_end > metadata.param_start:
        param_direction = 1
        direction = 1
    else:
        param_direction = -1
        direction = -1

    X_ref = X0.copy()
    X_old = X0.copy()
    z_ref = np.zeros_like(X0)
    z_ref[-1] = 1

    Dscale = np.ones_like(X0)
    Dscale_prev = Dscale.copy()

    # Initial step
    Xp, J, nfev, success = solve_nonlinear_system(X0, X_ref, z_ref, Dscale, Parametrisation.Local, ds, motion, metadata.dims)
    if not success:
        exit(1)
    X0 = Xp.copy()
    X_mat[:, 0] = X0

    det_jac_old = np.linalg.det(J[:-1, :-1])
    det_jac_ext_old = np.linalg.det(J)
    det_jac = det_jac_old
    det_jac_ext = det_jac_ext_old

    iteration = 1
    while iteration < metadata.max_steps:
        if metadata.scaling:
            Dscale_prev = Dscale.copy()
            Dscale = np.maximum(np.abs(X0 * Dscale_prev), np.ones_like(X0))
            Dscale[-2] = 1.0 # omega is not scaled
            Dscale[-1] = 1.0 # param is not scaled
            X_ref = X_ref * (Dscale_prev / Dscale)
            X0 = X0 * (Dscale_prev / Dscale)
            X_old = X_old * (Dscale_prev / Dscale)

        # J = extended_residual_jacobian(X0, X_ref, z_ref * Dscale_prev, hb_residual, Dscale, Parametrisation.ArcLength, True, False)
        ztmp = tangent_predictor(J @ np.diag(1 / Dscale_prev), z_ref * Dscale_prev, X_ref) / Dscale
        z = ztmp / np.linalg.norm(ztmp)

        # Take a step in the tangent direction ensuring to stay along the solution path
        if (iteration > 1) and np.dot(X0-X_old, direction*ds*z) < 0:
            direction *= -1

        # Parametrizaton params
        X_ref = X0.copy()
        z_ref = z.copy()

        # Predictor step
        Xp = X0 + direction*ds*z
        # Corrector step
        Xtmp, J, nfev, success = solve_nonlinear_system(Xp, X_ref, z_ref, Dscale, Parametrisation.ArcLength, ds, motion, metadata.dims)
        if not success:
            if ds > ds_min:
                print(f"Nonlinear solver failed, reducing step size from {ds:.3f} to {ds/2:.3f}")
                ds /= 2.0
                continue
            else:
                print("Nonlinear solver failed, continuation stopped")
                break
        det_jac = np.linalg.det(J[:-1, :-1])
        det_jac_ext = np.linalg.det(J)

        if det_jac * det_jac_old < 0 and det_jac_ext * det_jac_ext_old > 0:
            metadata.fold_pt.append(iteration)
            print("!!! Fold point detected !!!")
        elif det_jac * det_jac_old < 0 and det_jac_ext * det_jac_ext_old < 0:
            print("!!! Branch point detected !!!")
            metadata.branch_pt.append(iteration)

        det_jac_old = det_jac
        det_jac_ext_old = det_jac_ext

        X_old = X0.copy()
        X0 = Xtmp.copy()

        if metadata.step_adapt:
            xi = nfev_opt / nfev 
            xi = np.clip(xi, 0.5, 2.0)
            ds = np.clip(ds * xi, ds_min, ds_max)
            print(f"ds: {ds:.3f}")
        metadata.ds.append(ds)

        # History
        X_mat[:, iteration] = X0 * Dscale

        plot_hb(X_mat[:, iteration], iteration, metadata.dims)

        iteration += 1
        
        if (X0[-1] - metadata.param_end) * param_direction >= 0:
            print("Continuation reached the end")
            break

    metadata.X = X_mat[:, :iteration]
    filename = f"continuation_{hash_metadata(metadata)}.pkl"
    print(f"Continuation finished, saving to {filename}")
    with open(f"build/{filename}", 'wb') as f:
        pickle.dump(metadata, f)

    return metadata

def plot_hb_continuation(metadata):
    dofs = metadata.dims.n_d
    print(f"dofs: {dofs}")
    
    fig = plot.fig_create(dofs+1, 1, tuple([f"DOF {i+1}" for i in range(dofs)] + ["Omega"]), "Continuation")

    X_mat_h = metadata.X[:-2, :]
    for dof in range(dofs):
        X_h = X_mat_h[dof::dofs, :]
        for h in range(1, metadata.dims.n_h+1):
            fig.add_trace(
                go.Scatter(
                    x = metadata.X[-1, :],
                    y = np.sqrt(X_h[2*h-1, :]**2 + X_h[2*h, :]**2),
                    name = f"Harmonic {h}",
                    mode = "lines+markers"
                ),
                row=dof+1,
                col=1
            )
        plot.format_subplot(fig, dof+1, 1, r"$\bar{U}$", f"$||H_{dof+1}||^{2}$")
    fig.add_trace(
        go.Scatter(
            x = metadata.X[-1, :],
            y = metadata.X[-2, :],
            mode = "lines+markers"
        ),
        row=dofs+1,
        col=1
    )
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
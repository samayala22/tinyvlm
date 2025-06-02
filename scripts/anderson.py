import numpy as np
import scipy as sp
import scipy.optimize as opt
import scipy.linalg.lapack as lapack
import plotly.express as px

class FunctionCounter:
    def __init__(self, func):
        self.func = func
        self.eval_count = 0

    def __call__(self, x):
        self.eval_count += 1
        return self.func(x)
    
    def reset(self):
        self.eval_count = 0

def anderson_acceleration_fast(f, x0, k_max=100, tol_res=1e-6, m=3):
    """
    Compute a fixed point of f using Anderson Acceleration with fixed-allocated buffers.

    code

    Here we preallocate two buffers:
    Xbuf : an (n x m) array of differences in x
    Gbuf : an (n x m) array of differences in residuals, where g = f(x) - x
    
    We “initialize” the process with two fixed-point steps so that we have one difference available.
    Then each accelerated iteration uses the available differences with
        gamma = argmin || g_curr - Gbuf * gamma ||
    and the accelerated update is
        x_new = x_curr + g_curr - (Xbuf + Gbuf) * gamma.

    When the number of stored differences reaches m, we simply shift the buffers
    “to the left” and insert the newest differences in the last column.

    Parameters:
    f      : Function mapping R^n to R^n.
    x0     : Initial guess (n-dimensional numpy array).
    k_max  : Maximum number of iterations.
    tol_res: Tolerance on the residual ||f(x) - x||.
    m      : Maximum history (memory) used for acceleration.
    
    Returns:
    x_curr : Computed fixed point (as a numpy array).
    k      : Number of iterations performed.
    """
    n = x0.shape[0]
    # Allocation
    Xbuf = np.zeros((n, m))  # differences in iterates
    Gbuf = np.zeros((n, m))  # differences in residuals (g = f(x) - x)
    x_curr = np.zeros(n)
    x_new = np.zeros(n)
    g_curr = np.zeros(n)
    g_new = np.zeros(n)
    gamma = np.zeros(n)
    residual_history = []

    # Initialization
    # Let x_prev = x0 and compute one fixed–point step.
    x_curr = x0.copy() # in reality x_curr actually initialzed to x0
    x_new = f(x_curr)              # x₁ = f(x₀)
    g_curr = x_new - x_curr     # g₀ = f(x₀) - x₀
    Xbuf[:, 0] = g_curr # x1 - x0
    x_curr = x_new.copy()
    # Do one more fixed–point update to get a second residual
    x_new = f(x_curr)              # x₂ = f(x₁)
    g_new = x_new - x_curr        # g₁ = f(x₁) - x₁
    Gbuf[:, 0] = g_new - g_curr # g1 - g0
    g_curr = g_new.copy()
    
    # --- Anderson acceleration iterations ---
    k = 1
    while k < k_max and np.linalg.norm(g_curr) > tol_res:
        m_k = min(m, k)
        # Solve the least–squares problem:
        #     gamma = argmin || g_curr - Gbuf[:, :m_k]*gamma ||
        # gamma, _, _, _ = np.linalg.lstsq(Gbuf[:, :m_k], g_curr, rcond=None)
        _, gamma, _ = lapack.dgels(Gbuf[:, :m_k], g_curr)
        # gamma = sp.linalg.lstsq(Gbuf[:, :m_k], g_curr)[0]

        # Compute the update correction:
        # The acceleration step uses all stored differences (from both x and g):
        #    x_new = x_curr + g_curr - (Xbuf + Gbuf) * gamma.
        x_new = x_curr + g_curr - (Xbuf[:, :m_k] + Gbuf[:, :m_k]) @ gamma[:m_k]
        # Shift
        # if (m_k == m):
        #     Xbuf[:, :-1] = Xbuf[:, 1:]
        #     Gbuf[:, :-1] = Gbuf[:, 1:]

        # Xbuf[:, min(m_k, m-1)] = x_new - x_curr
        Xbuf[:, k % m] = x_new - x_curr
        x_curr = x_new.copy()
        x_new = f(x_curr)
        g_new = x_new - x_curr
        # Gbuf[:, min(m_k, m-1)] = g_new - g_curr
        Gbuf[:, k % m] = g_new - g_curr
        g_curr = g_new.copy()
        k += 1
        residual_history.append(np.linalg.norm(g_curr))

    # fig = px.line(x=range(k-1), y=residual_history, log_y=True)
    # fig.update_yaxes(
    #     tickformat='.2e',  # Format as exponential with 2 decimal places
    #     exponentformat='e'  # Use 'e' notation (alternatives: 'E', 'power', 'SI', 'B')
    # )
    # fig.show()

    return x_curr

def J(f, x, dx=1e-8):
    n = len(x)
    func = f(x)
    jac = np.zeros((n, n))
    for j in range(n):  # through columns to allow for vector addition
        Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
        x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
        jac[:, j] = (f(x_plus) - func)/Dxj
    return jac

def newton_krylov_anderson(f, x0, max_iter=100, tol_res=1e-6, m=5):
    delta_x = np.zeros(x0.shape) + 1.0
    x = x0.copy()
    iteration = 0
    # while np.linalg.norm(delta_x) > tol_res and iteration < max_iter:
    #     jac = J(f, x)
    #     delta_x = np.linalg.solve(jac, -f(x))
    #     x = x + delta_x
    #     iteration += 1

    # print(iteration)

    def inner(x):
        jac = J(f, x)
        delta_x = np.linalg.solve(jac, -f(x))
        return x + delta_x
    
    x = anderson_acceleration_fast(inner, x0)

    return x

def picard(f, x0, k_max=100, tol_res=1e-6):
    """Picard iteration for fixed-point problems.
       f: function mapping R^n to R^n
       x0: initial guess in R^n
       tol_res: tolerance for residual norm ||f(x) - x||_2 < tol_res
       k_max: maximum number of iterations
       """
    x_curr = x0.copy()
    k = 0
    while k < k_max and np.linalg.norm(f(x_curr) - x_curr) > tol_res:
        x_curr = f(x_curr)
        k += 1
    return x_curr

def mapping_2d(x):
    return np.array([2.0 * np.sin(x[0]) + np.arctan(x[1]),
                     np.sin(x[1]) + 2.0 * np.arctan(x[0])]) - x

def nonlinear_fixed_point(n=100, scale=0.9, bias_scale=0.1, seed=42):
    np.random.seed(seed)
    A_temp = np.random.randn(n, n)
    spectral_norm = np.linalg.norm(A_temp, 2)
    A = (scale / spectral_norm) * A_temp
    b = bias_scale * np.random.randn(n)
    delta=0.5
    
    def h(x):
        return np.tanh(A @ x + b) + 0.1 * np.sin(A @ x + b)
    
    def f(x):
        return x + delta * (h(x) - x)

    return f

def linear_system(n):
    np.random.seed(5)
    A = np.random.rand(n, n)
    b = np.random.rand(n)
    alpha = 0.5

    # Check spectral properties
    max_eig = np.max(np.abs(np.linalg.eigvals(A)))
    if max_eig >= 1 - alpha:
        print(f"Warning: Max eigenvalue of A ({max_eig}) is >= (1-alpha) ({1-alpha}).")
        print("Fixed-point iterations will likely diverge.")

    return lambda x: A @ x - b - x + alpha*x

def create_fixed_point(func):
    def f(x):
        return func(x) + x
    
    return f

def validate(func, sol, tol=1e-5):
    return "SUCCESS" if (np.all(np.abs(func(sol)) < tol)) else "FAILURE"

def main():
    k_max = 500
    tol_res = 1e-6
    m = 5

    func = nonlinear_fixed_point(100)
    x0 = np.random.rand(100)

    rf = FunctionCounter(func)
    fp = FunctionCounter(create_fixed_point(func))

    ok = validate(func, anderson_acceleration_fast(fp, x0, k_max, tol_res, m))
    print(f"[{ok}] Anderson Function evaluations: {fp.eval_count}")
    fp.reset()

    # ok = validate(func, picard(fp, x0, k_max, tol_res))
    # print(f"[{ok}] Picard Function evaluations: {fp.eval_count}")
    # fp.reset()

    ok = validate(func, opt.newton_krylov(rf, x0, method='lgmres', verbose=0, maxiter=k_max, f_tol=tol_res))
    print(f"[{ok}] Newton-Krylov Function evaluations: {rf.eval_count}")
    rf.reset()

    ok = validate(func, newton_krylov_anderson(rf, x0))
    print(f"[{ok}] Newton-Krylov Anderson Function evaluations: {rf.eval_count}")
    rf.reset()

    ROOT_METHODS = ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson']
    for method in ROOT_METHODS:
        try:
            ok = validate(func, opt.root(rf, x0, method=method, tol=tol_res).x)
            print(f"[{ok}] Scipy {method} Function evaluations: {rf.eval_count}")
            rf.reset()
        except Exception as e:
            print(f"[{False}] Scipy {method} failed: {e}")


if __name__ == '__main__':
    main()
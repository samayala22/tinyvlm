import numpy as np
from scipy.optimize import newton_krylov
import scipy.linalg.lapack as lapack
import time
import plotly.express as px

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
        # Compute the update correction:
        # The acceleration step uses all stored differences (from both x and g):
        #    x_new = x_curr + g_curr - (Xbuf + Gbuf) * gamma.
        x_new = x_curr + g_curr - (Xbuf[:, :m_k] + Gbuf[:, :m_k]) @ gamma[:m_k]
        # Shift
        if (m_k == m):
            Xbuf[:, :-1] = Xbuf[:, 1:]
            Gbuf[:, :-1] = Gbuf[:, 1:]

        Xbuf[:, min(m_k, m-1)] = x_new - x_curr
        x_curr = x_new.copy()
        x_new = f(x_curr)
        g_new = x_new - x_curr
        Gbuf[:, min(m_k, m-1)] = g_new - g_curr
        g_curr = g_new.copy()
        k += 1
        residual_history.append(np.linalg.norm(g_curr))

    fig = px.line(x=range(k-1), y=residual_history, log_y=True)
    fig.update_yaxes(
        tickformat='.2e',  # Format as exponential with 2 decimal places
        exponentformat='e'  # Use 'e' notation (alternatives: 'E', 'power', 'SI', 'B')
    )
    # fig.show()

    return x_curr, k

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
        if (k == k_max):
            print("Warning: Picard iteration did not converge.")
    return x_curr, k

def mapping_2d(x):
    return np.array([2.0 * np.sin(x[0]) + np.arctan(x[1]),
                     np.sin(x[1]) + 2.0 * np.arctan(x[0])])

# def f(x):
#     return np.array([np.sin(x[0]) + np.arctan(x[0])])

def nonlinear_fixed_point(n=100, scale=0.9, bias_scale=0.1, seed=42):
    np.random.seed(seed)
    A_temp = np.random.randn(n, n)
    spectral_norm = np.linalg.norm(A_temp, 2)
    A = (scale / spectral_norm) * A_temp
    b = bias_scale * np.random.randn(n)
    delta=0.2
    
    def h(x):
        return np.tanh(A @ x + b) + 0.1 * np.sin(A @ x + b)
    
    def f(x):
        return x + delta * (h(x) - x)

    return h

def bench(f: callable, func: callable, warmup=10, tries=50):
    for i in range(warmup):
        sol = f()[0]
    start = time.time()
    for i in range(tries):
        sol = f()[0]
    end = time.time()
    print("Avg time: ", (end - start) / tries)
    # assert np.allclose(func(sol), sol, atol=1e-6)

class FunctionCounter:
    def __init__(self, func):
        self.func = func
        self.eval_count = 0

    def __call__(self, x):
        self.eval_count += 1
        return self.func(x)
    
    def reset(self):
        self.eval_count = 0

def main():
    # Initial guess for the 2D fixed point.
    # x0 = np.array([1.5, 1.5])
    # x0 = np.array([1.0])
    np.random.seed(42)  # For reproducibility

    # n = 5
    # x0 = np.random.rand(n)
    # func = nonlinear_fixed_point(n)

    x0 = np.array([1.0, 1.0])
    func = mapping_2d

    k_max = 1000
    tol_res = 1e-6
    m = 2

    def F(x): return func(x)-x
    ff = FunctionCounter(F)
    fff = FunctionCounter(func)
    fixed_point, iterations = anderson_acceleration_fast(fff, x0, k_max, tol_res, m)
    print("Anderson Function evaluations:", fff.eval_count)
    print(fixed_point)
    # print(f"Anderson computed fixed point after {iterations} iterations")
    assert np.allclose(func(fixed_point), fixed_point, atol=tol_res)

    fff.reset()
    fixed_point, iterations = picard(fff, x0, k_max, tol_res)
    print("Picard Function evaluations:", fff.eval_count)

    # print(f"Picard computed fixed point after {iterations} iterations")
    # assert np.allclose(func(fixed_point), fixed_point, atol=tol_res)

    krylov_solution = newton_krylov(ff, x0, method='lgmres', verbose=0, maxiter=5000, f_tol=1e-6)
    print("Newto-Krylov Function evaluations:", ff.eval_count)
    # assert np.allclose(func(krylov_solution), krylov_solution, atol=1e-6)
    # print("Newton-Krylov satisfies the fixed point condition.")
    # bench(lambda: picard(func, x0, k_max, tol_res), func)
    # bench(lambda: anderson_acceleration_fast(func, x0, k_max, tol_res, m), func)
    # bench(lambda: newton_krylov(lambda x: func(x)-x, x0, method='lgmres', verbose=0, maxiter=5000, f_tol=1e-6), func)

if __name__ == '__main__':
    main()
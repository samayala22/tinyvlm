import numpy as np

EPS_f = np.float32(1.19209e-07)
EPS_sqrt_f = np.sqrt(EPS_f)

a = np.float32(0.1)
b = np.float32(0.75)

def func(t: np.float32):
    return a * np.sin(b * t)

def analytical_derivative(t: np.float32):
    return a * b * np.cos(b * t)

def derivative(f, t):
    # h = np.max((np.sqrt(t) * EPS_sqrt_f, EPS_f))
    h = EPS_sqrt_f
    return (f(t + h) - f(t - h)) / (2 * h)

def complex_step_derivative(f, t, h=EPS_sqrt_f):
    return np.imag(f(t + h*1j)) / h

n = 500
vec_t = np.linspace(0, 100, n)

abs_err_fdm = [np.abs(derivative(func, t) - analytical_derivative(t)) for t in vec_t]
abs_err_csd = [np.abs(complex_step_derivative(func, t) - analytical_derivative(t)) for t in vec_t]

print(f"Avg. abs. error (FDM): {np.mean(abs_err_fdm):.3e}")
print(f"Avg. abs. error (CSD): {np.mean(abs_err_csd):.3e}")
import numpy as np

EPS = np.finfo(np.float64).eps

def fd_h(x0): return np.maximum(np.sqrt(EPS) * np.abs(x0), EPS)

def cd_h(x0): return np.maximum(np.cbrt(EPS) * np.abs(x0), EPS)
def cd2_h(x0): return np.maximum(np.where(np.abs(x0) > 1, np.cbrt(EPS * np.abs(x0)), np.cbrt(EPS) * np.abs(x0)), EPS)
# def cd2_h(x0): return np.maximum(np.where(np.abs(x0) > 1, (1+np.log10(np.abs(x0))) * np.cbrt(EPS), np.cbrt(EPS) * np.abs(x0)), EPS)

def fd(f, x0, h=None):
    if h is None: h = fd_h(x0)
    return (f(x0 + h) - f(x0)) / h

def fd2(f, x0, h=None):
    if h is None: h = fd_h(x0)
    return (f(x0 + h) - f(x0)) / (x0 + h - x0)

def cd(f, x0, h=None):
    if h is None: h = cd_h(x0)
    return (f(x0 + h) - f(x0 - h)) / (2 * h)

def cd2(f, x0, h=None):
    if h is None: h = cd2_h(x0)
    return (f(x0 + h) - f(x0 - h)) / ((x0 + h) - (x0 - h))

def numerical_jac(f, x, *args, method="2-point"):
    m = len(f(x, *args))
    n = len(x)

    jac = np.zeros((m, n))
    if method == "3-point":
        for j in range(n):
            h = cd2_h(x[j])
            # print(f"{j}| h: {h:.5e}")
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            delta = xp[j] - xm[j] # delta representable fp number
            jac[:, j] = (f(xp, *args) - f(xm, *args)) / delta
    elif method == "2-point":
        yj = f(x, *args)
        for j in range(n):
            h = fd_h(x[j])
            xp = x.copy()
            xp[j] += h
            delta = xp[j] - x[j]
            jac[:, j] = (f(xp, *args) - yj) / delta
    
    return jac
    
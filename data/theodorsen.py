import numpy as np
import matplotlib.pyplot as plt

# RK4 (dx/ds = f(x, s))
def rk4(x0: float, s0: float, sf: float, ds: float, f: callable):
    s = np.arange(s0, sf, ds)
    x = np.zeros(len(s))
    x[0] = x0

    for i in range(1, len(s)):
        k1 = ds * f(x[i-1], s[i-1])
        k2 = ds * f(x[i-1] + 0.5 * k1, s[i-1] + 0.5 * ds)
        k3 = ds * f(x[i-1] + 0.5 * k2, s[i-1] + 0.5 * ds)
        k4 = ds * f(x[i-1] + k3, s[i-1] + ds)
        x[i] = x[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return s, x

# Two point central difference
def derivative(f, x):
    EPS_sqrt_f = np.sqrt(1.19209e-07)
    return (f(x + EPS_sqrt_f) - f(x - EPS_sqrt_f)) / (2 * EPS_sqrt_f)

# Jone approximation of Wagner function
b0 = 1
b1 = -0.165
b2 = -0.335
beta_1 = 0.0455
beta_2 = 0.3

# UVLM parameters
u_inf = 1 # freestream
b = 5 # wing total span
a = 1 # wing chord

def pitch(t): return 0
def heave(t): return np.sin(0.2 * t)

# Define the function w(s)
def w(s: float): return u_inf * pitch(s) + derivative(heave, s) + b * (0.5 - a) * derivative(pitch, s)

def dx1ds(x1: callable, s: float): return b1 * beta_1 * w(s) - beta_1 * x1(s)
def dx2ds(x2: callable, s: float): return b2 * beta_2 * w(s) - beta_2 * x2(s)

# # Initial condition
# x0 = # Your initial value for x_aug1 at s = s0
# s0 = # Your start value for s
# sf = # Your end value for s
# ds = # Your step size for s

# # Solve the differential equation
# s, x_aug1 = rk4(x0, s0, sf, ds)

# # Plot the solution
# plt.plot(s, x_aug1)
# plt.xlabel('s')
# plt.ylabel('x_aug1(s)')
# plt.title('Solution of the differential equation')
# plt.show()
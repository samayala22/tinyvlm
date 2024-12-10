import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

EPS_sqrt_f = np.sqrt(1.19209e-07)

# Two point central difference
def derivative(f, x):
    return (f(x + EPS_sqrt_f) - f(x - EPS_sqrt_f)) / (2 * EPS_sqrt_f)

def derivative2(f, x):
    return (f(x + EPS_sqrt_f) - 2 * f(x) + f(x - EPS_sqrt_f)) / (EPS_sqrt_f ** 2)

def solve_ivp(x0: float, s0: float, sf: float, f: callable):
    return spi.solve_ivp(f, [s0, sf], [x0]).y[-1] # return only the result at t=sf

def cl_theodorsen1(t: np.ndarray, alpha:callable, h:callable, a_h: float, b: float, u_inf: float):
    """
    t: time
    alpha: angle of attack
    h: heave
    a_h: dimensionless distance from center chord to pitch axis
    returns cl vector
    """
    # Jone's approximation of Wagner function
    b0 = 1
    b1 = -0.165
    b2 = -0.335
    beta_1 = 0.0455
    beta_2 = 0.3

    rho = 1.0
    c = 2*b

    def w(s: float): return u_inf * alpha(s) + derivative(h, s) + b * (0.5 - a_h) * derivative(alpha, s)
    def dx1ds(s: float, x1: float): return (u_inf / b) * (b1 * beta_1 * w(s) - beta_1 * x1)
    def dx2ds(s: float, x2: float): return (u_inf / b) * (b2 * beta_2 * w(s) - beta_2 * x2)
    x1_solution = spi.solve_ivp(dx1ds, [0, t[-1]], [0], t_eval=t)
    x2_solution = spi.solve_ivp(dx2ds, [0, t[-1]], [0], t_eval=t)
    def x1(s: float): return np.interp(s, x1_solution.t, x1_solution.y[0])
    def x2(s: float): return np.interp(s, x2_solution.t, x2_solution.y[0])
    L_m = rho * b * b * np.pi * (u_inf * derivative(pitch, t) + derivative2(heave, t) - b * a_h * derivative2(pitch, t))
    L_c = -2 * np.pi * rho * u_inf * b * (-(b0 + b1 + b2) * w(t) + x1(t) + x2(t))
    return (L_m + L_c) / (0.5 * rho * u_inf * u_inf * c)

# 4 augmented variables
def cl_theodorsen2(t: np.ndarray, alpha:callable, h:callable, a_h: float):
    """
    t: dimensionlesstime
    alpha: angle of attack
    h: dimensionlessheave
    a_h: dimensionless distance from center chord to pitch axis
    returns cl vector
    """
    psi1 = 0.165
    psi2 = 0.335
    eps1 = 0.0455
    eps2 = 0.3
    def dw1ds(s: float, w1: float): return alpha(s) - eps1 * w1
    def dw2ds(s: float, w2: float): return alpha(s) - eps2 * w2
    def dw3ds(s: float, w3: float): return h(s) - eps1 * w3
    def dw4ds(s: float, w4: float): return h(s) - eps2 * w4
    w1_solution = spi.solve_ivp(dw1ds, [0, t[-1]], [0], t_eval=t)
    w2_solution = spi.solve_ivp(dw2ds, [0, t[-1]], [0], t_eval=t)
    w3_solution = spi.solve_ivp(dw3ds, [0, t[-1]], [0], t_eval=t)
    w4_solution = spi.solve_ivp(dw4ds, [0, t[-1]], [0], t_eval=t)
    def w1(s: float): return np.interp(s, w1_solution.t, w1_solution.y[0])
    def w2(s: float): return np.interp(s, w2_solution.t, w2_solution.y[0])
    def w3(s: float): return np.interp(s, w3_solution.t, w3_solution.y[0])
    def w4(s: float): return np.interp(s, w4_solution.t, w4_solution.y[0])
    part0 = (alpha(t) - alpha(0) + derivative(h, t) - derivative(h, 0) + (0.5 - a_h)*(derivative(alpha, t) - derivative(alpha, 0)))
    part11 = -psi1*(alpha(t) - np.exp(-eps1*t)*alpha(0) - eps1*w1(t))
    part12 = -psi1*(derivative(h, t) - np.exp(-eps1*t)*derivative(h, 0) - eps1*h(t) + eps1 * np.exp(-eps1*t)*h(0) +eps1**2*w3(t))
    part13 = -psi1*(0.5-a_h)*(derivative(alpha, t) - np.exp(-eps1*t)*derivative(alpha, 0) - eps1*alpha(t) + eps1 * np.exp(-eps1*t)*alpha(0) + eps1**2*w1(t))
    part21 = -psi2*(alpha(t) - np.exp(-eps2*t)*alpha(0) - eps2*w2(t))
    part22 = -psi2*(derivative(h, t) - np.exp(-eps2*t)*derivative(h, 0) - eps2*h(t) + eps2 * np.exp(-eps2*t)*h(0) +eps2**2*w4(t))
    part23 = -psi2*(0.5-a_h)*(derivative(alpha, t) - np.exp(-eps2*t)*derivative(alpha, 0) - eps2*alpha(t) + eps2 * np.exp(-eps2*t)*alpha(0) + eps2**2*w2(t))
    cl = np.pi*(derivative2(h, t) - a_h * derivative2(alpha, t) + derivative(alpha, t)) + 2*np.pi*(alpha(0) + derivative(h, 0) + (0.5 - a_h)*derivative(alpha, 0))*(1-psi1*np.exp(-eps1*t)-psi2*np.exp(-eps2*t))+2*np.pi*(part0+part11+part12+part13+part21+part22+part23)
    return cl

# 2 augmented variables
def cl_theodorsen3(t: np.ndarray, alpha:callable, h:callable, a_h: float):
    """
    t: dimensionlesstime
    alpha: angle of attack
    h: dimensionlessheave
    a_h: dimensionless distance from center chord to pitch axis
    returns cl vector
    """
    psi1 = 0.165
    psi2 = 0.335
    eps1 = 0.0455
    eps2 = 0.3
    def w(s: float): return derivative(alpha, s) + derivative2(h, s) + (0.5 - a_h) * derivative2(alpha, s)
    def dw1ds(s: float, w1: float): return w(s) - eps1 * w1
    def dw2ds(s: float, w2: float): return w(s) - eps2 * w2
    w1_solution = spi.solve_ivp(dw1ds, [0, t[-1]], [0], t_eval=t)
    w2_solution = spi.solve_ivp(dw2ds, [0, t[-1]], [0], t_eval=t)
    def w1(s: float): return np.interp(s, w1_solution.t, w1_solution.y[0])
    def w2(s: float): return np.interp(s, w2_solution.t, w2_solution.y[0])
    part0 = (alpha(t) + derivative(h, t) + (0.5 - a_h)*derivative(alpha, t))
    part1 = -psi1*w1(t)
    part2 = -psi2*w2(t)
    # duhamel = u[1, i] - u[1, 0] + v[0, i] - v[0, 0] + (0.5 - ndv.a_h)*(v[1, i] - v[1, 0]) - psi1*w1 - psi2*w2
    # wagner_init = (u[1, 0] + u[0, 0] + (0.5 - ndv.a_h)*v[1,0])*(1 - psi1*np.exp(-eps1*t) - psi2*np.exp(-eps2*t))
    duhamel = alpha(t) - alpha(0) + derivative(h, t) - derivative(h, 0) + (0.5 - a_h)*(derivative(alpha, t) - derivative(alpha, 0)) - psi1*w1(t) - psi2*w2(t)
    wagner_init = (alpha(0) + derivative(h, 0) + (0.5 - a_h)*derivative(alpha, 0))*(1 - psi1*np.exp(-eps1*t) - psi2*np.exp(-eps2*t))
    # cl = np.pi*(derivative2(h, t) - a_h * derivative2(alpha, t) + derivative(alpha, t)) + 2*np.pi*(alpha(0) + derivative(h, 0) + (0.5 - a_h)*derivative(alpha, 0))*(-psi1*np.exp(-eps1*t)-psi2*np.exp(-eps2*t))+2*np.pi*(part0+part1+part2)
    # cl = np.pi*(a[0, i] - ndv.a_h * a[1, i] + v[1, i]) + 2*np.pi*(wagner_init + duhamel)

    cl = np.pi*(derivative2(h, t) - a_h * derivative2(alpha, t) + derivative(alpha, t)) + 2*np.pi*(wagner_init + duhamel)
    cm = np.pi*(0.5 + a_h)*(wagner_init + duhamel) + 0.5*np.pi*a_h*(derivative2(h, t) - a_h*derivative2(alpha, t)) - 0.5*np.pi*(0.5 - a_h)*derivative(alpha, t) - (np.pi/16) * derivative2(alpha, t)
    return cl, cm

if __name__ == "__main__":
    # UVLM parameters
    rho = 1 # fluid density
    u_inf = 1.0 # freestream
    ar = 10000 # aspect ratio
    b = 0.5 # half chord
    c = 2*b # chord
    a = ar / c # full span
    a_h = -0.5 # quarter chord

    nb_pts = 1000
    k = 0.5 # reduced frequency
    t_final_nd = 90 # nd time
    omega = k * u_inf / b # frequency
    vec_t_nd = np.linspace(0, t_final_nd, nb_pts)

    # sudden acceleration
    # def pitch(t): return np.radians(2.8)
    # def heave(t): return 0

    # pure heaving
    # def pitch(t): return 0
    # def heave(t): return -0.1 * np.sin(omega * t)

    # pure pitching
    def pitch(t): return np.radians(np.sin(omega * t))
    def heave(t): return 0

    fig, axs = plt.subplot_mosaic(
        [["cl"], ["cm"]],  # Disposition des graphiques
        constrained_layout=True,  # Demander à Matplotlib d'essayer d'optimiser la disposition des graphiques pour que les axes ne se superposent pas
        figsize=(11, 6),  # Ajuster la taille de la figure (x,y)
    )

    # axs["time"].plot(vec_t, cl_theodorsen1(vec_t, pitch, heave, a_h, b), label=f"Theodorsen (k={k})")
    # vec_t_nd = u_inf * vec_t / b # non-dimensional time
    # axs["time"].plot(vec_t, cl_theodorsen3(vec_t_nd, lambda t: pitch(t*b / u_inf), lambda t: heave(t*b / u_inf) / b, a_h), label=f"Theodorsen 2 (k={k})")

    def plot_uvlm(axs_cl, axs_cm, filename, label):            
        uvlm_t_nd = []
        uvlm_cl = []
        uvlm_cm = []
        with open(filename, "r") as f:
            _ = float(f.readline())
            for line in f:
                t, cl, cm = map(float, line.split())
                uvlm_t_nd.append(t)
                uvlm_cl.append(cl)
                uvlm_cm.append(cm)
        
        axs_cl.plot(uvlm_t_nd, uvlm_cl, label=label)
        axs_cm.plot(uvlm_t_nd, uvlm_cm, label=label)

    plot_uvlm(axs["cl"], axs["cm"], "build/windows/x64/debug/2dof_aero.txt", "UVLM")

    vec_cl, vec_cm = cl_theodorsen3(vec_t_nd, pitch, heave, a_h)

    axs["cl"].plot(vec_t_nd, vec_cl, label="Theodorsen")    
    axs["cm"].plot(vec_t_nd, vec_cm, label="Theodorsen")
    
    axs["cl"].set_xlabel('t')
    axs["cl"].set_ylabel('CL')
    axs["cl"].grid(True, which='both', linestyle=':', linewidth=1.0, color='gray')
    axs["cl"].legend()

    axs["cm"].set_xlabel('t')
    axs["cm"].set_ylabel('CM')
    axs["cm"].grid(True, which='both', linestyle=':', linewidth=1.0, color='gray')
    axs["cm"].legend()

    plt.show()

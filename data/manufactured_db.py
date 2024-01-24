import numpy as np
import matplotlib.pyplot as plt

def erf(x : np.ndarray) -> np.ndarray:
    # formula 7.1.26 from Handbook of Mathematical Functions (Abramowitz and Stegun)
    # https://coewww.rutgers.edu/~norris/bookmarks_files/abramowitz_and_stegun.pdf
    # save the sign of x
    sign_mask = x < 0
    x = np.abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    # Invert sign erf(-x) = -erf(x)
    y[sign_mask] *= -1
    return y 

class CL:
    def __init__(self, cl_0, a0, a1, cl_a0, cl_a1):
        self.cl_0 = cl_0
        self.a0 = a0
        self.a1 = a1
        self.cl_a0 = cl_a0
        self.cl_a1 = cl_a1
    def generate(self, x: np.ndarray):
        return self.cl_a0 * x + 0.5 * (self.cl_0 - self.cl_a1 * x) * (1 + erf((x - self.a0) / self.a1))

alphas = np.radians(np.linspace(0, 30))
cl_factory = CL(1.2, 0.28, 0.02, 2*np.pi, 2*np.pi)
cls = cl_factory.generate(alphas)

plt.plot(alphas, cls)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$C_L$')
plt.show()
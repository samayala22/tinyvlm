import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
import matplotlib as mpl
from finite_diff import *

def create_functions(scale):
    # def f(x): return scale * np.sin(3*x) * np.log(x)
    # def df(x): return scale*(3*np.cos(3*x) * np.log(x) + np.sin(3*x) / x)
    # def f(x): return scale * np.sin(x) * np.log(1 + x**2)
    # def df(x): return scale * (np.cos(x) * np.log(1 + x**2) + (2*x*np.sin(x))/(1 + x**2))
    # def f(x): return scale * (x**2 + x - 1.34)
    def f(x): return scale * (np.sin(x)*np.cos(3*x))
    # def f(x): return scale * (np.exp(x) / (np.sqrt(np.sin(x**3) + np.cos(x**3))))
    def df(x): return egrad(f)(x)
    return f, df

def step_sweep(methods, h_methods, f, df, x0: float):
    h = np.logspace(-15, -1, num=300)
    # h = np.logspace(-50, -1, num=300, base=2)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, method in enumerate(methods):
        err = np.abs(method(f, x0, h) - df(x0))
        plt.loglog(h, err, 'o-', markersize=3, label=method.__name__)
    
    for i, h_method in enumerate(h_methods):
        h_val = h_method(x0)
        plt.axvline(x=h_val, linestyle='--', color = next(ax._get_lines.prop_cycler)['color'], linewidth=i+1, label=h_method.__name__)

    plt.xlabel('Step size (h)')
    plt.ylabel('Absolute error')
    plt.title('Derivative Error vs. Step Size')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()

def value_sweep(methods, f, df):
    assert len(methods) == 4 # limitation for plotting
    n_x = 100
    n_scale = 100
    x = np.logspace(-5, 5, n_x)
    f_scales = np.logspace(-5, 5, n_x)
    err = np.zeros((n_scale, n_x, len(methods)))
    for i, scale in enumerate(f_scales):
        f, df = create_functions(scale)
        for m, method in enumerate(methods):
            err[i, :, m] = np.abs(method(f, x) - df(x))
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    images = []
    # norm = LogNorm(vmin=1e-18, vmax=1e5, clip=True) # Broken so we have to normalize manually
    norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=True)
    for i in range(2):
        for j in range(2):
            lidx = i*2 + j
            normalized_err = (16+np.log10(np.minimum(err[:, :, lidx], 1e-1))) / 16
            im = axs[i, j].imshow(normalized_err, cmap='plasma', norm=norm)
            images.append(im)
            axs[i, j].set_title(methods[lidx].__name__)
            axs[i, j].invert_yaxis()
     
    fig.subplots_adjust(right=0.85)  # Make space for colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    fig.colorbar(images[0], cax=cbar_ax, label='Error')
    plt.show()

    for i, method in enumerate(methods):
        print(f"{method.__name__} Avg Error: {np.average(err[:, :, i])} | Median Error: {np.median(err[:, :, i])}")

if __name__ == '__main__':
    f, df = create_functions(100.0)

    methods = [fd, fd2, cd, cd2]
    h_methods = [fd_h, cd_h, cd2_h]

    step_sweep(methods, h_methods, f, df, 100.0)
    value_sweep(methods, f, df)
    
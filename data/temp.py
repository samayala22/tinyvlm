import numpy as np

def X_to_complex(X):
    """
    Converts a dofs * (2H+1) real array [A0, A1, B1, ... A_H, B_H] to a dofs * (H+1) comlex array
    """
    assert len(X.shape) == 2 # matrix form
    dofs = X.shape[0]
    H = int((X.shape[1] - 1) / 2)
    Xc = np.zeros((dofs, H+1), dtype=np.complex128)
    for d in range(dofs):
        Xc[d, 0] = X[d, 0] - 0j
    for h in range(1, H+1):
        for d in range(dofs):
            Xc[d, h] = (X[d, 2*h-1] - 1j * X[d, 2*h])/2
    return Xc

def X_to_real(X):
    """
    Converts a dofs * N complex array to a dofs * (2*N-1) real array
    """
    assert len(X.shape) == 2 # matrix form
    dofs = X.shape[0]
    N = X.shape[1]
    L = np.ones(N)
    Xr = np.zeros((dofs, 2*N-1), dtype=np.float64)
    for d in range(dofs):
        Xr[d, 0] = X[d, 0].real
    for h in range(1, N):
        for d in range(dofs):
            Xr[d, 2*h-1] = L[h] * 2 * X[d, h].real
            Xr[d, 2*h] = L[h] * -2 * X[d, h].imag
    
    return Xr

hbvlm_coeffs = np.load("build/windows/x64/release/hbvlm.npy")
hbvlm_coeffs_t = np.load("build/windows/x64/release/hbvlm_t.npy")

N = hbvlm_coeffs.shape[1]
H = (N-1)//2
hbvlm_coeffs = hbvlm_coeffs.astype(np.float64)
hbvlm_coeffs_t = hbvlm_coeffs_t.astype(np.float64)
hbvlm_coeffs[0]  *= np.sqrt(1 / N)
hbvlm_coeffs[1:] *= np.sqrt(2 / N)
hbvlm_coeffs_complex = X_to_complex(hbvlm_coeffs)
q = np.fft.irfft(hbvlm_coeffs_complex, N, axis=1, norm='forward') # no scaling

hbvlm_coeffs2 = np.fft.rfft(hbvlm_coeffs_t, N, axis=1, norm='backward') # no scaling
hbvlm_coeffs2 = X_to_real(hbvlm_coeffs2[:, :H+1] / N)
q2 = np.fft.irfft(X_to_complex(hbvlm_coeffs2), N, axis=1, norm='forward') # no scaling

print(hbvlm_coeffs)
print(hbvlm_coeffs2)

gt = np.array([
    [0.19405, 0.252967, 0.0455411, 0.179768, 0.332238, 0.0207069, -0.289905, -0.13586, -0.0773092, -0.332283, -0.235856],
    [-0.0141577, 0.0632901, 0.0432724, 0.024986, 0.071175, 0.0577027, -0.0350097, -0.0632378, -0.0244726, -0.046118, -0.0814341]
])

print(np.linalg.norm(q - gt))
print(np.abs(q - gt))

print(np.linalg.norm(q2 - gt))
print(np.abs(q2 - gt))

print("cl: ", q[0, :])
print("cm: ", q[1, :])

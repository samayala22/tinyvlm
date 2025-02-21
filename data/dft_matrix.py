import numpy as np

if __name__ == "__main__":

    harmonics = 3
    unknowns = 2 * harmonics + 1
    omega = 0.5
    period = 2.0 * np.pi / omega

    dft = np.zeros((unknowns, unknowns))
    dft[:, 0] = 1.0
    for j in range(1, unknowns, 2):
        k = int((j - 1) / 2) + 1
        for i in range(0, unknowns):
            tn = (i / unknowns) * period
            dft[i, j] = np.cos(omega * tn * k)
            dft[i, j + 1] = np.sin(omega * tn * k)

    # Normalize so that inv(D) = D^T
    dft[:, 0] /= np.sqrt(unknowns)
    dft[:, 1:] *= np.sqrt(2 / unknowns)

    print(dft)
    print(np.allclose(np.linalg.inv(dft), dft.T))
    print(np.allclose(np.linalg.inv(dft.T), dft))
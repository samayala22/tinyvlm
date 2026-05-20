import numpy as np

if __name__ == "__main__":

    harmonics = 3
    unknowns = 2 * harmonics + 1
    omega = 0.5
    period = 2.0 * np.pi / omega

    dft = np.zeros((unknowns, unknowns))

    sqrt_unknowns = 1 / np.sqrt(unknowns)
    sqrt_unknowns_2 = np.sqrt(2 / unknowns)
    
    for i in range(unknowns): # first col
        dft[i, 0] = sqrt_unknowns
    for j in range(1, unknowns, 2):
        k = (j + 1) / 2
        for i in range(0, unknowns):
            tn = (i / unknowns) * period
            dft[i, j] = np.cos(omega * tn * k) * sqrt_unknowns_2
            dft[i, j + 1] = np.sin(omega * tn * k) * sqrt_unknowns_2

    print(dft)
    print(np.allclose(np.linalg.inv(dft), dft.T))
    print(np.allclose(np.linalg.inv(dft.T), dft))
import numpy as np
import matplotlib.pyplot as plt

def diffusion_matrix(N, dx):
    M = np.zeros((N, N), dtype=np.float64)

    for i in range(1, N-1):
        M[i, i-1], M[i, i], M[i, i+1] = dx, -2.0*dx, dx
    
    M[0, 0], M[0, 1] = -2.0*dx, dx
    M[N-1, N-2], M[N-1, N-1] = dx, -2.0*dx

    return M

def dirichlet_solve(N, dx, T0, T1):
    RHS = np.zeros(N, dtype=np.float64)

    RHS[0] = -dx*T0
    RHS[N-1] = -dx*T1

    M = diffusion_matrix(N, dx)
    T = np.linalg.solve(M, RHS)
    return T

if __name__ == "__main__":
    T = dirichlet_solve(100, 1.0 / 101, 10, 30)
    plt.plot(T)
    plt.show()

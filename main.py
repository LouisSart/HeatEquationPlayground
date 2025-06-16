import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def diffusion_matrix(N, left="dirichlet", right="dirichlet"):
    M = np.zeros((N, N), dtype=np.float64)

    for i in range(1, N-1):
        M[i, i-1], M[i, i], M[i, i+1] = 1.0, -2.0, 1.0
    
    if left == "dirichlet":
        M[0, 0], M[0, 1] = -2.0, 1.0
    elif left == "neumann":
        M[0, 0], M[0, 1] = -1.0, 1.0
    else:
        raise ValueError("Invalid boundary condition")
    if right == "dirichlet":
        M[N-1, N-2], M[N-1, N-1] = 1.0, -2.0
    elif right == "neumann":
        M[N-1, N-2], M[N-1, N-1] = 1.0, -1.0
    else:
        raise ValueError("Invalid boundary condition")

    return M

def iteration_matrix(N, D, dx, dt, left="dirichlet", right="dirichlet"):
    I = np.eye(N, dtype=np.float64)
    M = diffusion_matrix(N, left, right)
    return I + dt * D * M / (dx * dx)


def rhs(N, D, dx, dt, left = ("dirichlet", 0), right = ("dirichlet", 0)):
    phi = np.zeros(N, dtype=np.float64)
    if left[0] == "dirichlet":
        phi[0] = dt * D * left[1] / dx**2
    elif left[0] == "neumann":
        phi[0] = dt * D * left[1] / dx
    else:
        raise ValueError("Invalid boundary condition")
    if right[0] == "dirichlet":
        phi[N-1] = dt * D * right[1] / dx**2
    elif right[0] == "neumann":
        phi[N-1] = dt * D * right[1] / dx
    else:
        raise ValueError("Invalid boundary condition")
    return phi

def initial_condition(N, Tmax):
    T0 = np.zeros(N, dtype=np.float64)
    for k in range(N):
        T0[k] = Tmax*np.sin(k * np.pi / (N - 1))
    return T0

def cfl(D, dx):
    return dx**2 / (2 * D)

def loop(T0, M, rhs, dt, tmax):
    T = [T0.copy()]
    t = 0.0
    i = 1
    while t < tmax:
        # print(i, t, tmax)
        T.append(M @ T[i-1] + rhs)
        t += dt
        i += 1
    return T


if __name__ == "__main__":
    N = 100
    L = 1.0
    dx = L / (N - 1)
    D = 1.0
    dt = 0.9 * cfl(D, dx)
    left = ("neumann", 0.0)
    right = ("neumann", 0.0)
    M = iteration_matrix(N, D, dx, dt, left[0], right[0])
    rhs = rhs(N, D, dx, dt, left, right)

    T0 = initial_condition(N, 30.0)

    T = loop(T0, M, rhs, dt, 0.5)
    plt.plot(T0)
    plt.plot(T[1], label="t=0")
    plt.plot(T[100], label=f"t={100*dt}")
    plt.plot(T[500], label=f"t={500*dt}")
    plt.plot(T[1000], label=f"t={1000*dt}")
    plt.plot(T[10000], label=f"t={10000*dt}")
    plt.legend()
    plt.show()

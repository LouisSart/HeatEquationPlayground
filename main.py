import numpy as np
from anim import *

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
        T0[k] = Tmax * (np.sin(k * np.pi / (N - 1)))
    return T0

def initial_condition_square(N, Tmax):
    T0 = np.zeros(N, dtype=np.float64)
    for k in range(N):
        if k > N / 3 and k < 2 * N / 3:
            T0[k] = Tmax
    return T0

def cfl(D, dx):
    return dx**2 / (2 * D)

def loop(T0, M, rhs, dt, tmax):
    T = [T0.copy()]
    t = 0.0
    i = 1
    while t < tmax:
        T.append(M @ T[i-1] + rhs)
        t += dt
        i += 1
    return T


if __name__ == "__main__":
    # Aluminium :
    lbd = 273 # W.m-1.K-1
    rho = 2700 # kg.m-3
    cp = 897 # J.kg-1.K-1
    D = lbd / (rho * cp) # m2.s-1

    # Space and time discretization
    N = 100
    L = 1.0
    dx = L / (N - 1)
    dt = 0.9 * cfl(D, dx)

    # Boundary conditions
    left = ("neumann", 0.0)
    right = ("neumann", 0.0)
    M = iteration_matrix(N, D, dx, dt, left[0], right[0])
    rhs = rhs(N, D, dx, dt, left, right)

    T0 = initial_condition(N, 30.0)

    T = loop(T0, M, rhs, dt, 300)
    ani = animate(T, len(T), 200)
    ani.save(filename="sin_right_neu_left_neu.gif", writer="pillow", fps=30)
    # plt.show()

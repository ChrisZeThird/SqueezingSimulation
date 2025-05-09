import numpy as np
import matplotlib.pyplot as plt

from utils.misc import factorial


def S(N, kappa, xi):
    """
    Computes the integral term int_0^xi exp(-kappa tau)/(1+tau^2) dt
    :param N: order of the series expansion
    :param kappa: the absorption coefficient as defined in BK
    :param xi: focusing parameter as defined in BK
    :return:
    """
    s_n = 0
    for n in range(N):
        fac_n = factorial(n)
        a_n = ((-1) ** n) * (2 * fac_n)

        s_p = 0
        for p in range(2*n+1):
            fac_p = factorial(p)
            common_factor = np.exp(-kappa * xi) + (-1)**(p + 1) * np.exp(kappa * xi)
            s_p += common_factor * (xi ** p) / (fac_p * (kappa ** (2 * n + 1 - p)))

        temp = a_n * (s_p + 2/((2 * n + 1) * kappa ** (2 * n + 2)))
        s_n += temp

    return s_n


def h(kappa, xi, N):
    """
    Boyd-Kleinman function h
    :param kappa:
    :param xi:
    :param N:
    :return:
    """
    s_n = S(N, kappa, xi)
    return 1/(2 * xi) * (s_n ** 2)


# Parameters
xi_vals = np.linspace(0.1, 100, 300)
kappa_vals = [0]
N_vals = [20, 50, 80]

plt.figure(figsize=(12, 8))

for kappa in kappa_vals:
    for N in N_vals:
        h_vals = [h(kappa, xi, N) for xi in xi_vals]
        plt.plot(xi_vals, h_vals, label=f"kappa={kappa}, N={N}")

plt.title("Boyd-Kleinman function h(xi) for different kappa and N")
plt.xlabel("xi (focusing parameter)")
plt.ylabel("h(kappa, xi, N)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




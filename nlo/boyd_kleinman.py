import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


def integrand1(t, kappa):
    return np.exp(-kappa * t) / (1 + t**2)


def integrand2(t, kappa):
    return (np.exp(-kappa * t) / (1 + t**2)) * t


def compute_integral(integrand, kappa, xi):
    result, error = quad(integrand, -xi, xi, args=(kappa,))
    return result


# Parameters
kappa = 0.
xi_vals = np.linspace(start=0.1, stop=100, num=1000)

# Compute I and h for each xi
I_vals_1 = np.array([compute_integral(integrand1, kappa, xi) for xi in xi_vals])
I_vals_2 = np.array([compute_integral(integrand2, kappa, xi) for xi in xi_vals])
h_vals = (1 / (4 * xi_vals)) * (I_vals_1 ** 2 + I_vals_2 ** 2)

h_exact = (1 / xi_vals) * np.arctan(xi_vals)**2

plt.plot(xi_vals, h_vals, label=f'Numerical h(κ=0)')
plt.plot(xi_vals, h_exact, '--', label='Exact h(κ=0) = arctan²(ξ)/ξ')
plt.xlabel('ξ')
plt.ylabel('h(κ, ξ)')
plt.title('Comparison of Numerical and Exact h(κ=0, ξ)')
plt.grid(True)
plt.legend()
plt.show()
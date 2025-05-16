import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from scipy.constants import c, pi, epsilon_0
from scipy.integrate import dblquad
from scipy.optimize import minimize_scalar

from utils.settings import settings

import utils.plot_parameters as pm

# ---------------------------
# KTP absorption
# ---------------------------
alpha1, alpha2 = 0.01, 0.01


# ---------------------------
# Core Functions
# ---------------------------

def Xi(crystal_length, waist, wavelength, index=settings.crystal_index):
    b = (2 * pi * waist**2) / wavelength  # confocal parameter
    xi = crystal_length / b
    return xi


def WaistFromXi(xi, crystal_length, wavelength, index):
    b = crystal_length / xi
    return np.sqrt((b * wavelength) / (2 * pi))


def compute_K(alpha1, alpha2, waist, wavelength):
    b = (2 * np.pi * waist**2) / wavelength
    alpha = alpha1 + 0.5 * alpha2
    K = 0.5 * alpha * b
    return K


def integrand(tau_prime, tau, sigma, K):
    exponent = -K * (tau + tau_prime) + 1j * sigma * (tau - tau_prime)
    denominator = (1 + 1j * tau) * (1 - 1j * tau_prime)
    return (np.exp(exponent) / denominator).real  # Only real part contributes to SHG power


def compute_F(sigma, xi, K):
    result, _ = dblquad(
        lambda tau_prime, tau: integrand(tau_prime, tau, sigma, K),
        -xi, xi,
        lambda tau: -xi, lambda tau: xi,
        epsabs=1e-6, epsrel=1e-6,  # tighter tolerance
    )
    return result / (4 * pi**2)


def compute_F_fast(sigma, xi, K, N=500):  # increase N
    tau = np.linspace(-xi, xi, N)
    tau_prime = np.linspace(-xi, xi, N)
    d_tau = tau[1] - tau[0]

    T, T_prime = np.meshgrid(tau, tau_prime)
    exponent = -K * (T + T_prime) + 1j * sigma * (T - T_prime)
    denominator = (1 + 1j * T) * (1 - 1j * T_prime)
    integrand = (np.exp(exponent) / denominator).real

    integral = np.sum(integrand) * d_tau**2
    return integral / (4 * pi**2)


def compute_h(sigma, xi, K):
    # F_val = compute_F(sigma, xi, K)
    F_val = compute_F_fast(sigma, xi, K)
    return (pi**2 / xi) * F_val


def optimize_hm(xi, K):
    res = minimize_scalar(lambda sigma: -compute_h(sigma, xi, K), bounds=(-10, 10), method='bounded')
    return -res.fun, res.x  # Return maximum h and optimal sigma


# ---------------------------
# Generate curves for different K values
# ---------------------------

xi_vals = np.logspace(start=-2, stop=1.2, num=70)  # range of xi values
K_values = [0.0, 0.15, 0.3]     # absorption parameters
results = {}

for K in K_values:
    h_vals = []
    for xi in xi_vals:
        h, _ = optimize_hm(xi, K)
        h_vals.append(h)
    h_vals = np.array(h_vals)
    max_idx = np.argmax(h_vals)
    xi_opt = xi_vals[max_idx]
    h_max = h_vals[max_idx]
    waist_opt = WaistFromXi(xi_opt, settings.crystal_length, settings.wavelength, settings.crystal_index)
    results[K] = {
        "xi_vals": xi_vals,
        "h_vals": h_vals,
        "xi_opt": xi_opt,
        "waist_opt": waist_opt,
        "h_max": h_max
    }

# ---------------------------
# Plotting
# ---------------------------

plt.figure(figsize=(10, 6))

for K, data in results.items():
    label = (rf"$K={K}$: "
             rf"$\xi_{{\mathrm{{opt}}}}={data['xi_opt']:.2f},\, "
             rf"w_0={data['waist_opt']*1e6:.1f}\,\mu m,\, "
             rf"h_{{\max}}={data['h_max']:.3f}$")
    plt.plot(data["xi_vals"], data["h_vals"], label=label)

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r"$\xi$", fontsize=14)
plt.ylabel(r"$h_m(\xi)$", fontsize=14)
plt.title("Boyd–Kleinman Optimized $h_m(\\xi)$ for Different Absorption $K$", fontsize=15)

plt.grid(True, which="both", linestyle="--", alpha=0.6)

plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

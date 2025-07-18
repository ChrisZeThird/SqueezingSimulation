import matplotlib.pyplot as plt
import numpy as np

from scipy.constants import c, epsilon_0
from scipy.integrate import dblquad
from scipy.optimize import fsolve

from nlo.sellmeier import n_z
import utils.plot_parameters
from utils.settings import settings

# Analysis of singly resonant SHG efficiency versus focusing
wavelength_pump = settings.wavelength
wavelength_harmonic = wavelength_pump / 2

omega = 2 * np.pi * settings.c / wavelength_pump

d_eff = 9.5e-12  # effective nonlinear coefficient for PPKTP
n_pump = n_z(wavelength_pump * 1e6)  # convert to micrometers for n_z function
n_harmonic = n_z(wavelength_harmonic * 1e6)  # convert to micrometers for n_z function

alpha_harmonic = 0.14  # absorption coefficient at 2ω (cm^-1)
alpha_harmonic /= 100  # convert to m^-1
alpha_pump = 0  # assume negligible FF absorption

L_c = settings.crystal_length  # crystal length in meters
k_omega = 2 * np.pi * n_pump / wavelength_pump


# Function for h(α, L, r), here r = 0
def h_function(alpha, L):
    def integrand(s, sp):
        numerator = np.exp(-alpha * (s + sp + L))
        denominator = (1 + 1j * s) * (1 - 1j * sp)
        return (numerator / denominator).real

    val, _ = dblquad(integrand, -L / 2, L / 2, lambda s: -L / 2, lambda s: L / 2)
    return val / (2 * L)


# Focusing parameter sweep
w0_range = np.linspace(15e-6, 150e-6, 500)
z_R = np.pi * w0_range ** 2 * n_pump / wavelength_pump
L_array = L_c / z_R
alpha = (alpha_pump - alpha_harmonic / 2) * z_R

# Compute C_effective for each w0
Gamma_eff = []
for a, L in zip(alpha, L_array):
    h_val = h_function(a, L)
    c_eff = (2 * omega ** 2 * d_eff ** 2 / (np.pi * epsilon_0 * c ** 3 * n_pump ** 2 * n_harmonic)) \
           * L_c * k_omega * np.exp(-alpha_harmonic * L_c) * h_val
    Gamma_eff.append(c_eff)

Gamma_eff = np.array(Gamma_eff)  # convert to numpy array for further calculations
Gamma_tot = 1.1 * Gamma_eff  # total gain coefficient, plane wave approximation from Le Targat et al. 2011


# Optimal transmission factor
def T1_opt(Gamma, epsilon=0.02, P_in=310e-3):
    """
    Calculate the optimal transmission factor T1.
    Default values from Le Targat et al. 2011.
    :param Gamma: gain coefficient
    :param epsilon: small parameter for stability
    :param P_in: input power
    :return: optimal transmission factor T1
    """
    return (epsilon / 2) + np.sqrt((epsilon / 2) ** 2 + (Gamma * P_in))


# Eq (3): Solve numerically for eta
def solve_eta(T1, Gamma, Gamma_eff, P_in=310e-3, epsilon=0.02):
    """
    Solve the implicit eta equation from Le Targat et al.:
    sqrt(eta) * [2 - sqrt(1 - T1) * (2 - epsilon - Gamma * sqrt(eta * P_in / Gamma_eff))]^2
    - 4 * T1 * sqrt(Gamma_eff * P_in) = 0
    """
    def equation(eta):
        if eta < 0 or eta > 1:
            return np.inf  # restrict to physical domain
        sqrt_eta = np.sqrt(eta)
        sqrt_term = np.sqrt(1 - T1)
        inner = 2 - epsilon - Gamma * np.sqrt((eta * P_in) / Gamma_eff)
        bracket = 2 - sqrt_term * inner
        lhs = sqrt_eta * bracket**2
        rhs = 4 * T1 * np.sqrt(Gamma_eff * P_in)
        return lhs - rhs

    eta_guess = 0.5
    eta_solution, = fsolve(equation, eta_guess)
    return eta_solution if 0 <= eta_solution <= 1 else np.nan


# Input power range
P_in_values = np.array([100, 200, 310, 500])  # Different input powers in watts
eta_results = {}

for P_in in P_in_values:
    eta_vals = []
    T1 = T1_opt(Gamma=Gamma_tot, P_in=P_in*1e-3)
    for T1, G, Geff in zip(T1, Gamma_tot, Gamma_eff):
        eta = solve_eta(T1, G, Geff, P_in=P_in*1e-3)
        eta_vals.append(eta)
    eta_results[str(P_in)] = np.array(eta_vals)


# Plotting
plt.figure(figsize=(8, 5))
for P_in, eta_vals in eta_results.items():
    plt.plot(L_array, eta_vals, label=f"P_in = {P_in} mW")
plt.xlabel("Focusing Parameter L = Lc / zR")
plt.ylabel("Conversion Efficiency η")
plt.title("SHG Efficiency η vs Focusing Parameter L for Different Input Powers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

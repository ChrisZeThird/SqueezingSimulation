from numpy import cosh, tanh, sqrt, zeros, exp, linspace
from math import factorial
import matplotlib.pyplot as plt


def squeezed_vacuum_state(r, dim):
    """Return the squeezed vacuum state |zeta> as a vector of length 'dim'."""
    vec = zeros(dim, dtype=complex)
    norm_factor = 1.0 / sqrt(cosh(r))
    for n in range(dim // 2):
        k = 2 * n
        coeff = (sqrt(factorial(2 * n)) / (2**n * factorial(n))) * (tanh(r))**n
        vec[k] = norm_factor * coeff
    return vec


def squeezed_vacuum_photon_prob(r, n_max):
    """
    Calculate the photon number probability distribution for a squeezed vacuum state.
    Only even photon numbers have nonzero probability.

    :param r: Squeezing parameter (real).
    :param n_max: Maximum photon number (should be even).

    Returns: probs, List of probabilities P_n for n = 0, ..., n_max.
    """
    probs = []
    norm = cosh(r)
    for n in range(n_max + 1):
        if n % 2 == 0:
            m = n // 2
            prob = (factorial(2 * m) / (2 ** (2 * m) * (factorial(m) ** 2))) * (tanh(r) ** (2 * m)) / norm
            probs.append(prob)
        else:
            probs.append(0.0)
    return probs


# Parameters
r = 1.0      # Squeezing parameter
n_max = 20   # Maximum photon number to plot

# Compute probabilities
probs = squeezed_vacuum_photon_prob(r, n_max)

# Plot
plt.figure(figsize=(8, 4))
plt.bar(range(n_max+1), probs, color='royalblue')
plt.xlabel('Photon number n')
plt.ylabel('Probability P(n)')
plt.title(f'Photon number distribution for squeezed vacuum (r={r})')
plt.grid(axis='y', alpha=0.3)
plt.show()


# Parameters for the Gaussian squeezing spectrum
omega0 = 2.0      # Center frequency (arbitrary units, e.g., 2*omega)
delta = 0.5       # Bandwidth (standard deviation)
S0 = -3.0         # Maximum squeezing in dB (negative means noise reduction)

# Frequency axis
omega = linspace(omega0 - 2*delta, omega0 + 2*delta, 500)


# Gaussian squeezing spectrum (in dB)
def squeezing_spectrum(omega, omega0, delta, S0):
    return S0 * exp(- (omega - omega0)**2 / (2 * delta**2))


S = squeezing_spectrum(omega, omega0, delta, S0)

# Plotting
plt.figure(figsize=(7,4))
plt.plot(omega, S, label='Squeezing Spectrum')
plt.axhline(0, color='gray', linestyle='--', label='Shot noise')
plt.xlabel(r'Frequency $\omega$')
plt.ylabel('Squeezing (dB)')
plt.title('Gaussian Squeezing Spectrum around $2\\omega_0$')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

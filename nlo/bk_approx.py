import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.constants import c, pi
from scipy.special import erf
from utils.settings import settings


# Define the functions as before

def Xi(crystal_length, waist, wavelength, index):
    b = (2 * pi * waist**2)/wavelength  # confocal parameter
    xi = crystal_length / b
    return xi


def Kappa(B, xi):
    delta = 2 * B * np.sqrt(xi)
    numerator = np.exp(-delta ** 2) - 1 + np.sqrt(pi) * delta * erf(delta)
    return numerator / (delta ** 2)


def c1(B):
    return 0.876 - (18.8/(B + 36.5)) + (0.0166 / (0.0693 + (B - 0.440)**2)) - (0.283/(0.931 + (B + 0.516)**3))


def c2(B):
    return 0.530 - (36.0/(B + 95.1)) + (0.0103 / (0.332 + (B - 0.569)**2)) - (0.497/(4.69 + (B + 1.15)**3))


def c3(B):
    return 0.796 - (0.506/(B + 0.378)) + (0.0601 / (0.421 + (B - 0.673)**2)) + (0.0329/(0.0425 + (B - 0.221)**3))


def hm(xi, kappa=1, B=0):
    numerator = np.arctan(c1(B) * kappa * xi)
    denominator = c1(B) + c2(B) * xi * np.arctan(c3(B) * xi)
    return numerator / denominator


# Plot setup
log_xi = np.linspace(-3, 3, 500)
xi_vals = 10 ** log_xi
B_values = [1, 2, 4, 8]
waist_default = 22.6e-6  # Default waist value
xi_default = Xi(settings.crystal_length, waist_default, wavelength=settings.wavelength, index=settings.crystal_index)


# Plotting function to update the plot based on waist

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
# Add slider for waist adjustment
ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
waist_slider = Slider(ax_slider, 'Waist', 10e-6, 300e-6, valinit=waist_default, valstep=0.05e-6)


def update(val):
    waist = waist_slider.val
    xi = Xi(settings.crystal_length, waist, wavelength=settings.wavelength, index=settings.crystal_index)

    # Clear the current plot
    ax.clear()

    # Plot the curves for B values
    ax.plot(log_xi, hm(xi_vals, kappa=1, B=0), label="B = 0")
    for walkoff in B_values:
        kappa_vals = Kappa(walkoff, xi_vals)
        hm_vals = hm(xi_vals, kappa_vals, walkoff)
        ax.plot(log_xi, hm_vals, label=f"B = {walkoff}")

    # Add the vertical line at the specific xi
    ax.axvline(x=np.log10(xi), color='k', linestyle='--', label=f"$\\xi$ = {xi:.2e}")

    # Mark the intersection points and display the values of hm
    for walkoff in [0] + B_values:
        if walkoff != 0:
            kappa_vals = Kappa(walkoff, xi)
        else:
            kappa_vals = 1

        hm_val_at_xi = hm(xi, kappa=kappa_vals, B=walkoff)
        ax.scatter(np.log10(xi), hm_val_at_xi, color='black', zorder=5)
        ax.text(np.log10(xi) + 0.05, hm_val_at_xi, f"{hm_val_at_xi:.2e}", color='black', fontsize=10, verticalalignment='bottom')

    ax.set_xlabel(r"$\log_{10}(\xi)$")
    ax.set_ylabel(r"$h_m(\xi)$")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e1)
    ax.set_title(r"$h_m(\xi)$ vs $\log_{10}(\xi)$ for different $B$")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.canvas.draw_idle()


# Initial plot update
update(None)


# Set slider update function
waist_slider.on_changed(update)

# Show the plot with slider
plt.show()

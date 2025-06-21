import numpy as np
import matplotlib.pyplot as plt
from cavity.cavity_formulas import stability_condition
from utils.settings import settings

# Fixed parameters
l_crystal = settings.crystal_length  # Crystal length in meters
index_crystal = settings.crystal_index  # Refractive index of the crystal
wavelength = settings.wavelength  # Laser wavelength in meters

# Ranges for L and d_curved
L_range = np.linspace(start=0.5, stop=1.5, num=300)  # Cavity round-trip length in meters
d_curved_range = np.linspace(start=0.1, stop=0.4, num=300)  # Distance between curved mirrors in meters

# Fixed RoC values to compare
RoC_values = [100e-3, 150e-3, 200e-3]  # Radius of curvature in meters

# Plot stability condition for each RoC
fig, axes = plt.subplots(1, len(RoC_values), figsize=(18, 6), sharey=True)

for i, R in enumerate(RoC_values):
    stability_map = np.zeros((len(L_range), len(d_curved_range)), dtype=bool)
    waist_map = np.full_like(stability_map, np.nan, dtype=np.float64)

    for j, L in enumerate(L_range):
        for k, d_curved in enumerate(d_curved_range):
            valid_d_curved, stability_values, waist_values = stability_condition(
                d_curved=np.array([d_curved]),
                L=L,
                R=R,
                l_crystal=l_crystal,
                index_crystal=index_crystal,
                wavelength=wavelength
            )
            if valid_d_curved.size > 0:
                stability_map[j, k] = True
                waist_map[j, k] = waist_values[0]
                # print(waist_values)
        # break
    # Boolean plot for stability
    ax = axes[i]
    contour = ax.pcolormesh(d_curved_range * 1e3, L_range * 1e3, stability_map, shading='auto', cmap='viridis')
    ax.set_title(f"Stability Map for RoC = {R * 1e3:.0f} mm")
    ax.set_xlabel("Curved Mirror Distance $d_c$ (mm)")
    if i == 0:
        ax.set_ylabel("Cavity Length $L$ (mm)")

    # Overlay waist values
    waist_contour = ax.contour(d_curved_range * 1e3, L_range * 1e3, waist_map * 1e5, levels=5, colors='blue', linewidths=0.8)
    ax.clabel(waist_contour, inline=True, fontsize=8, fmt="%.2f µm")

# Adjust layout and show plot
plt.tight_layout()
plt.colorbar(contour, ax=axes, orientation='horizontal', label="Stability (True/False)")
plt.show()
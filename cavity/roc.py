import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from cavity.cavity_formulas import stability_condition
from utils.settings import settings

# Fixed parameters
l_crystal = settings.crystal_length
index_crystal = settings.crystal_index
wavelength = settings.wavelength

L_range = np.linspace(start=0.5, stop=0.9, num=800)
d_curved_range = np.linspace(start=0.1, stop=0.35, num=800)

RoC_values = [100e-3, 150e-3, 200e-3]

fig, axes = plt.subplots(1, len(RoC_values), figsize=(18, 6), sharey=True)

# Custom discrete colormap for True/False
cmap = ListedColormap(['white', 'green'])
norm = BoundaryNorm([0, 0.5, 1], cmap.N)

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

    ax = axes[i]
    contour = ax.pcolormesh(d_curved_range * 1e3, L_range * 1e3, stability_map, shading='auto', cmap=cmap, norm=norm)
    ax.set_title(f"Stability Map for RoC = {R * 1e3:.0f} mm")
    ax.set_xlabel("Curved Mirror Distance $d_c$ (mm)")
    if i == 0:
        ax.set_ylabel("Cavity Length $L$ (mm)")

    # Contour of beam waist (only over stable region)
    waist_masked = np.ma.masked_where(~stability_map, waist_map)
    # Only show contours in the 20–80 µm range
    levels = np.linspace(start=20, stop=60, num=3, endpoint=True)  # 7 evenly spaced levels between 20 and 80 µm
    waist_contour = ax.contour(
        d_curved_range * 1e3,
        L_range * 1e3,
        waist_masked * 1e6,  # Convert m → µm
        levels=levels,
        colors='black',
        linewidths=1.2,
        linestyles='dashed'
    )
    ax.clabel(waist_contour, inline=True, fontsize=8, fmt="%.1f µm")
    # After plotting pcolormesh and contours
    # Crop view to only stability region
    stable_indices = np.argwhere(stability_map)
    if stable_indices.size > 0:
        min_j, min_k = stable_indices.min(axis=0)
        max_j, max_k = stable_indices.max(axis=0)

        d_min = d_curved_range[min_k] * 1e3  # mm
        d_max = d_curved_range[max_k] * 1e3
        L_min = L_range[min_j] * 1e3
        L_max = L_range[max_j] * 1e3

        pad_d = (d_max - d_min) * 0.05
        pad_L = (L_max - L_min) * 0.05

        ax.set_xlim(d_min - pad_d, d_max + pad_d)
        ax.set_ylim(L_min - pad_L, L_max + pad_L)

plt.tight_layout()

# Discrete colorbar
cbar = fig.colorbar(contour, ax=axes, orientation='horizontal', ticks=[0.25, 0.75])
cbar.ax.set_xticklabels(['Unstable', 'Stable'])
cbar.set_label("Cavity Stability")

plt.show()

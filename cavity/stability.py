import numpy as np
import matplotlib.pyplot as plt

from cavity.cavity_formulas import ABCD_Matrix, stability_condition
from utils.settings import settings
import utils.plot_parameters

# Constants
L = settings.fixed_length
R = settings.R
d_curved = np.linspace(start=settings.d_curved_min, stop=settings.d_curved_max, num=settings.number_points)
l_crystal_array = np.array([10, 15, 20, 25, 30]) * 1e-3  # mm to meters

# Plotting stability parameter s vs d_curved for different crystal lengths
plt.figure(figsize=(10, 6))

for lc in l_crystal_array:
    d_valid, s_valid, w_valid = stability_condition(
        d_curved=d_curved,
        L=L,
        R=R,
        l_crystal=lc,
        index_crystal=settings.crystal_index,
        wavelength=settings.wavelength
    )

    if d_valid.size == 0:
        continue

    max_idx = np.argmax(w_valid)
    d_max = d_valid[max_idx] * 1e3  # mm
    s_max = s_valid[max_idx]
    w_max = w_valid[max_idx] * 1e6  # µm

    label = f'$l_{{c}}$ = {lc*1e3:.0f} mm, $w_{{max}}$ = {w_max:.1f} µm'
    line, = plt.plot(d_valid * 1e3, s_valid, label=label)
    color = line.get_color()

    # Optional: mark the point (not annotated anymore)
    plt.plot(d_max, s_max, 'o', color=color, markersize=6)

# Stability bounds
plt.axhline(1, color='r', linestyle='--', linewidth=1, label='Stability limits')
plt.axhline(-1, color='r', linestyle='--', linewidth=1)
plt.fill_between(d_curved * 1e3, -1, 1, color='green', alpha=0.1)

# Labels and layout
plt.xlabel('Curved mirror distance $d_c$ (mm)')
plt.ylabel('Stability parameter $s$')
plt.title('Stability parameter vs $d_c$ for different crystal lengths')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

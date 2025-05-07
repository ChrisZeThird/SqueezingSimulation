import numpy as np
import matplotlib.pyplot as plt

from cavity.cavity_formulas import ABCD_Matrix, stability_condition

from utils.settings import settings


# Constants (example values in mm)
L = settings.fixed_length
R = settings.R
d_curved = np.linspace(start=settings.d_curved_min, stop=settings.d_curved_max, num=settings.number_points)

l_crystal_array = np.array([10, 15, 20, 25, 30]) * 1e-3

# Compute stability for each l_crystal
s_values = []
dc = []

for lc in l_crystal_array:
    stable_region, dc_stable = stability_condition(d_curved, L, R, lc, index_crystal=settings.crystal_index,
                        wavelength=settings.wavelength)

    s_values.append(stable_region)
    dc.append(dc_stable)


# Plotting
plt.figure(figsize=(8, 5))
plt.plot(l_crystal_array, s_values, 'o-', label='Stability parameter s = (A + D)/2')

plt.axhline(1, color='r', linestyle='--', label='Stability bounds')
plt.axhline(-1, color='r', linestyle='--')
plt.fill_between(l_crystal_array, -1, 1, color='green', alpha=0.1, label='Stable region')

plt.xlabel('Crystal length $l_{crystal}$ (mm)')
plt.ylabel('Stability parameter $s$')
plt.title('Evolution of cavity stability with crystal length')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

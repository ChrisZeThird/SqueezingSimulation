import numpy as np
import matplotlib.pyplot as plt

from cavity.cavity_formulas import ABCD_Matrix, stability_condition

from utils.settings import settings


# Constants (example values in mm)
L = settings.fixed_length
R = settings.R
d_curved = settings.fixed_d_curved

l_crystal_array = np.array([10, 15, 20, 25, 30]) * 1e-3

# Compute stability for each l_crystal
stability_values = []
s_values = []

for lc in l_crystal_array:
    A1, B1, C1, D1 = ABCD_Matrix(L, d_curved, R, lc)
    # s = (2 * A1 * D1 - 1) / 2
    s = (2 * A1 * D1 - 1) / 2
    s_values.append(s)
    stability_values.append(-1 <= s <= 1)

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

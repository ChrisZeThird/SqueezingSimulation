import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils.settings import settings
from utils.misc import approximate_to_next_ten
import utils.plot_parameters as mplp

import cavity.cavity_formulas as cf

# Setting constants
c = settings.c

# Setting parameters
number_points = settings.number_points

min_L = settings.min_L
max_L = settings.max_L
min_T = settings.min_T
max_T = settings.max_T
cavity_lengths = np.linspace(start=min_L, stop=max_L, num=number_points)
transmission_coefficients = np.linspace(start=min_T, stop=max_T, num=number_points)

L, T = np.meshgrid(cavity_lengths, transmission_coefficients)

bandwidth_meshgrid = cf.Bandwidth_linear(cavity_length=L, transmission_coefficient=T) * 1e-6
clev = np.arange(bandwidth_meshgrid.min(), bandwidth_meshgrid.max(), 0.1)

# Find couple (L, T) such that Delta within range
central_freq = settings.central_freq  # MHz
threshold = settings.range_freq
boundary_down = central_freq - threshold * central_freq
boundary_up = central_freq + threshold * central_freq

# print(np.shape(bandwidth_meshgrid))
indices = np.where((bandwidth_meshgrid < 11) & (bandwidth_meshgrid > 9))   # here np.where gives a tuple, the first element of which gives the row index, while the
                                                                        # second element gives the corresponding column index
length_range = L[indices]
transmission_range = T[indices]

# Setting figure
fig, ax = plt.subplots()
contour_plot = ax.contourf(L, T, bandwidth_meshgrid, clev, cmap=settings.cmap_name)

# contour_plot = ax.imshow(bandwidth_meshgrid, cmap=cmap)

nb_ticks = 7
xticks = np.around(np.linspace(start=min_L, stop=max_L, num=nb_ticks), decimals=1)
yticks = np.around(np.linspace(start=min_T, stop=max_T, num=nb_ticks), decimals=1)
new_xticks = np.linspace(0, number_points, nb_ticks)
new_yticks = np.linspace(0, number_points, nb_ticks)

ax.set_xlabel('Cavity length L (m)')
ax.set_ylabel('Transmission coefficient')

# Highlight the region where bandwidth is within desired range

cbar = plt.colorbar(contour_plot, ax=ax)
cbar.outline.set_visible(False)

cbar.ax.axhline(y=central_freq, color='white', linewidth=50, alpha=settings.alpha)
ax.plot(length_range, transmission_range, c='white', alpha=settings.alpha)

cbar.set_label(r'Bandwidth $\Delta$ (MHz)')  # Set label

plt.show()

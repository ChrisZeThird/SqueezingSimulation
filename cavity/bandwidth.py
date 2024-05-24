import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils.settings import settings
from utils.misc import approximate_to_next_ten

# Setting constants
c = settings.c

# Setting parameters
number_points = 300

min_L = 0.5
max_L = 3.0
min_T = 0
max_T = 1
cavity_lengths = np.linspace(start=min_L, stop=max_L, num=number_points)
transmission_coefficients = np.linspace(start=min_T, stop=max_T, num=number_points)

L, T = np.meshgrid(cavity_lengths, transmission_coefficients)


# Defining function
def bandwidth(cavity_length, transmission_coefficient):
    """
    Calculates the bandwidth of the cavity from a couple (L, T)
    :param cavity_length: in meter
    :param transmission_coefficient:
    :return:
    """
    # return (c / (2 * cavity_length)) * (transmission_coefficient / (2 * np.pi))
    return (3e8 * transmission_coefficient) / (4 * np.pi * cavity_length)


bandwidth_meshgrid = bandwidth(cavity_length=L, transmission_coefficient=T) * 1e-6
clev = np.arange(bandwidth_meshgrid.min(), bandwidth_meshgrid.max(), 0.1)

# Find couple (L, T) such that Delta within range
central_freq = 6  # MHz
threshold = 0.1
boundary_down = central_freq - threshold * central_freq
boundary_up = central_freq + threshold * central_freq

# print(np.shape(bandwidth_meshgrid))
indices = np.where((bandwidth_meshgrid < 11) & (bandwidth_meshgrid > 9))   # here np.where gives a tuple, the first element of which gives the row index, while the
                                                                        # second element gives the corresponding column index
length_range = L[indices]
transmission_range = T[indices]

# -- Matplotlib parameters -- #
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

alpha = 0.3
# Set cmap
cmap = matplotlib.colormaps['plasma']  # define the colormap
# cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
# cmaplist[0] = (.5, .5, .5, 1.0)  # force the first color entry to be grey
#
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map
#
# max_bound = approximate_to_next_ten(np.amax(bandwidth_meshgrid))
# bounds = np.linspace(0, max_bound, 20)
# norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# Setting figure
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig, ax = plt.subplots()
contour_plot = ax.contourf(L, T, bandwidth_meshgrid, clev, cmap=cmap)

# contour_plot = ax.imshow(bandwidth_meshgrid, cmap=cmap)

nb_ticks = 7
xticks = np.around(np.linspace(start=min_L, stop=max_L, num=nb_ticks), decimals=1)
yticks = np.around(np.linspace(start=min_T, stop=max_T, num=nb_ticks), decimals=1)
new_xticks = np.linspace(0, number_points, nb_ticks)
new_yticks = np.linspace(0, number_points, nb_ticks)

# ax.set_xticks(new_xticks, labels=xticks)
# ax.set_yticks(new_yticks, labels=yticks)

ax.set_xlabel('Cavity length L (m)')
ax.set_ylabel('Transmission coefficient')

# Highlight the region where bandwidth is within desired range
# ax.contour(bandwidth_meshgrid, levels=[central_freq], colors='white')

cbar = plt.colorbar(contour_plot, ax=ax)
cbar.outline.set_visible(False)
# cbar.ax.axhline(y=boundary_down, color='white', linewidth=1, alpha=0.5)
# print(cbar.get_ticks())
# cbar.ax.axhline(y=boundary_up, color='white', linewidth=1, alpha=0.5)
cbar.ax.axhline(y=central_freq, color='white', linewidth=50, alpha=alpha)
ax.plot(length_range, transmission_range, c='white', alpha=alpha)
# Set label
# original_ticks = list(cbar.get_ticks())
# cbar.set_ticks(original_ticks + [central_freq])
# cbar.set_ticklabels(original_ticks + ['10'])

# ax_cbar = fig.add_axes([0.92, 0.1, 0.03, 0.8])  # create a second axes for the colorbar
# cbar = matplotlib.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm,
#     spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
# ax_cbar.set_ylabel(r'Bandwidth $\Delta$ (MHz)')
cbar.set_label(r'Bandwidth $\Delta$ (MHz)')

plt.show()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils.settings import settings
from utils.misc import approximate_to_next_ten

# Setting constants
c = settings.c

# Setting parameters
number_points = 1000

cavity_lengths = np.linspace(start=0.1, stop=1, num=number_points)
transmission_coefficients = np.linspace(start=0.5, stop=1, num=number_points)

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
clev = np.arange(bandwidth_meshgrid.min(), bandwidth_meshgrid.max(), 0.001)

# -- Matplotlib parameters -- #
SMALL_SIZE = 13
MEDIUM_SIZE = 20
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Set cmap
cmap = matplotlib.colormaps['plasma']  # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
cmaplist[0] = (.5, .5, .5, 1.0)  # force the first color entry to be grey

cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map

max_bound = approximate_to_next_ten(np.amax(bandwidth_meshgrid))
bounds = np.linspace(0, max_bound, 20)
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# Setting figure
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.contourf(L, T, bandwidth_meshgrid, clev, cmap=cmap)
ax.set_xlabel('Cavity length L (m)')
ax.set_ylabel('Transmission coefficient')

ax_cbar = fig.add_axes([0.92, 0.1, 0.03, 0.8])  # create a second axes for the colorbar
cbar = matplotlib.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm,
    spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
ax_cbar.set_ylabel(r'Bandwidth $\Delta$ (MHz)')
# cbar.set_label(r'Bandwidth $\Delta$ (MHz)')

plt.show()

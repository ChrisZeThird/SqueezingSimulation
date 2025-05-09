import matplotlib
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerBase
import matplotlib.pyplot as plt

import numpy as np

from utils.settings import settings

# -- Matplotlib parameters -- #
SMALL_SIZE = 18
MEDIUM_SIZE = 25
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

cmap = matplotlib.colormaps[settings.cmap_name]  # define the colormap


class AnyObjectHandler(HandlerBase):

    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0, y0 + width], [0.7 * height, 0.7 * height], linestyle=orig_handle[1], color=orig_handle[0])
        l2 = plt.Line2D([x0, y0 + width], [0.3 * height, 0.3 * height], color=orig_handle[0])
        return [l1, l2]


def symmetrical_colormap(cmap_settings, new_name=None):
    """
    This function take a colormap and create a new one, as the concatenation of itself by a symmetrical fold.
    :param cmap_settings:
    :param new_name:
    :return:
    """
    # get the colormap
    current_cmap = plt.cm.get_cmap(*cmap_settings)
    if not new_name:
        new_name = "sym_" + cmap_settings[0]  # ex: 'sym_Blues'

    # this defined the roughness of the colormap, 128 is fine
    n = 256

    # get the list of color from colormap
    colors_r = current_cmap(np.linspace(0, 1, n))  # take the standard colormap # 'right-part'
    colors_l = colors_r[::-1]  # take the first list of color and flip the order # "left-part"

    # combine them and build a new colormap
    colors = np.vstack((colors_r, colors_l))
    mymap = mcolors.LinearSegmentedColormap.from_list(new_name, colors)

    return mymap

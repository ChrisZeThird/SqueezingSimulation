import matplotlib
from matplotlib.legend_handler import HandlerBase
import matplotlib.pyplot as plt

from utils.settings import settings

# -- Matplotlib parameters -- #
SMALL_SIZE = 18
MEDIUM_SIZE = 25
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
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

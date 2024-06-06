import matplotlib.pyplot as plt
import numpy as np

import cavity.cavity_formulas as cf
from cavity.bandwidth import bandwidth
from cavity.waist import waist, Kaertner
from utils.settings import settings

if __name__ == '__main__':
    # -- OPTIMIZE bandwidth -- #
    if settings.plot_bandwidth:
        bandwidth()

    # -- Optimize waist -- #
    if settings.plot_waist:
        waist()

    # -- Kaertner class notes -- #
    if settings.plot_kaertner:
        Kaertner()

import matplotlib.pyplot as plt
import numpy as np

import cavity.cavity_formulas as cf
from cavity.bandwidth import bandwidth
from cavity.waist import waist, Kaertner

from squeezing.plot import squeezing_vs_pump, squeezing_vs_wavelength

from utils.settings import settings

if __name__ == '__main__':
    # -- Squeezing -- #
    if settings.plot_pump_power:
        squeezing_vs_pump()

    if settings.squeezing_wavelength:
        squeezing_vs_wavelength()

    # -- OPTIMIZE bandwidth -- #
    if settings.plot_bandwidth:
        bandwidth()

    # -- Optimize waist -- #
    if settings.plot_waist:
        waist()

    # -- Kaertner class notes -- #
    if settings.plot_kaertner:
        Kaertner()
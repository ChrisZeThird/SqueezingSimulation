import matplotlib.pyplot as plt
import numpy as np

import cavity.cavity_formulas as cf
from cavity.bandwidth import bandwidth
from cavity.waist import waist
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
        R1 = settings.R1
        R2 = settings.R2
        length_kaertner = np.linspace(start=settings.min_L, stop=settings.max_L, num=settings.number_points)

        w1 = cf.waist_mirror1(R1=settings.R1, R2=settings.R2, L=length_kaertner, wavelength=settings.wavelength)
        w2 = cf.waist_mirror3(R1=settings.R1, R2=settings.R2, L=length_kaertner, wavelength=settings.wavelength)
        w0 = cf.waist_intracavity(R1=settings.R1, R2=settings.R2, L=length_kaertner, wavelength=settings.wavelength)

        fig_waist_kaertner, ax_kaertner = plt.subplots(nrows=3, ncols=1)

        ax_kaertner[0].plot(length_kaertner * 1e2, w1 / np.sqrt(settings.wavelength * settings.R1 / np.pi), color='red')
        ax_kaertner[0].set_ylabel(r'$w_1 / (\lambda R_1 / \pi)^{1/2}$', fontsize=12)

        ax_kaertner[1].plot(length_kaertner * 1e2, w2 / np.sqrt(settings.wavelength * settings.R2 / np.pi), color='red')
        ax_kaertner[1].set_ylabel(r'$w_2 / (\lambda R_2 / \pi)^{1/2}$', fontsize=12)

        ax_kaertner[2].plot(length_kaertner * 1e2, w0 / np.sqrt(settings.wavelength / np.pi), color='red')
        ax_kaertner[2].set_ylabel(r'$w_0 / (\lambda / \pi)^{1/2}$', fontsize=12)

        ax_kaertner[2].set_xlabel('Cavity length L (cm)', fontsize=18)

        fig_waist_kaertner.tight_layout()
        plt.show()

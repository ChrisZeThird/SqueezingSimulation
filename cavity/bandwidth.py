import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils.settings import settings
import utils.plot_parameters as mplp

import cavity.cavity_formulas as cf


def bandwidth():
    # Setting parameters
    cavity_lengths = np.linspace(start=settings.min_L, stop=settings.max_L, num=settings.number_points)
    transmission_coefficients = np.linspace(start=settings.min_T, stop=settings.max_T, num=settings.number_points)

    L, T = np.meshgrid(cavity_lengths, transmission_coefficients)
    bandwidth_meshgrid = cf.Bandwidth_linear(cavity_length=L, transmission_coefficient=T) * 1e-6
    # bandwidth_meshgrid = cf.Bandwidth_bowtie(T=T, L=L, Loss=settings.cavity_loss) * 1e-6

    # Find couple (L, T) such that Delta within range
    boundary_down = settings.central_freq - settings.range_freq * settings.central_freq
    boundary_up = settings.central_freq + settings.range_freq * settings.central_freq
    indices = np.where((bandwidth_meshgrid < boundary_down) & (
                bandwidth_meshgrid > boundary_up))  # here np.where gives a tuple, the first element of which gives the row index, while the

    # gives row index, 2nd element gives column index
    length_range = L[indices]
    transmission_range = T[indices]

    # Plot bandwidth
    clev = np.arange(bandwidth_meshgrid.min(), bandwidth_meshgrid.max(), 0.2)

    fig_bandwidth, ax_bandwidth = plt.subplots()
    contour_plot = ax_bandwidth.contourf(L, T, bandwidth_meshgrid, clev, cmap=mplp.cmap)

    # Adjust ticks for colorbar
    nb_ticks = 7
    xticks_bandwidth = np.around(np.linspace(start=settings.min_L, stop=settings.max_L, num=nb_ticks), decimals=1)
    yticks_bandwidth = np.around(np.linspace(start=settings.min_T, stop=settings.max_T, num=nb_ticks), decimals=1)
    new_xticks_bandwidth = np.linspace(0, settings.number_points, nb_ticks)
    new_yticks_bandwidth = np.linspace(0, settings.number_points, nb_ticks)

    cbar = plt.colorbar(contour_plot, ax=ax_bandwidth)
    cbar.outline.set_visible(False)

    cbar.ax.axhline(y=settings.central_freq, color='white', linewidth=50, alpha=settings.alpha)
    ax_bandwidth.plot(length_range, transmission_range, c='white', alpha=settings.alpha)

    ax_bandwidth.set_xlabel('Cavity length L (m)')
    ax_bandwidth.set_ylabel('Transmission coefficient')
    ax_bandwidth.set_title(f'Intra-cavity loss: L = {settings.cavity_loss}')
    cbar.set_label(r'Bandwidth $\Delta$ (MHz)')

    plt.show()

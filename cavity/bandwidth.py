import matplotlib
from matplotlib.offsetbox import AnchoredText
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
    # bandwidth_meshgrid = cf.Bandwidth_linear(cavity_length=L, transmission_coefficient=T) * 1e-6
    bandwidth_meshgrid = cf.Bandwidth_bowtie(T=T, L=L, Loss=settings.cavity_loss) * 1e-6

    # Bandwidths list
    bandwidth_list = np.array([1, 10, 100]) * 1e6
    bandwidth_colours = ['lightsteelblue', 'cornflowerblue', 'royalblue']

    # Bandwidth surface plot
    clev = np.arange(bandwidth_meshgrid.min(), bandwidth_meshgrid.max(), 0.2)

    fig_bandwidth, ax_bandwidth = plt.subplots()
    contour_plot = ax_bandwidth.contourf(L, T, bandwidth_meshgrid, clev, cmap=mplp.cmap, norm=matplotlib.colors.LogNorm(vmin=1, vmax=bandwidth_meshgrid.max()))
    # contour_plot.set_clim(vmin=1, vmax=round(bandwidth_meshgrid.max()))

    # Adjust ticks for colorbar
    nb_ticks = 7
    xticks_bandwidth = np.around(np.linspace(start=settings.min_L, stop=settings.max_L, num=nb_ticks), decimals=1)
    yticks_bandwidth = np.around(np.linspace(start=settings.min_T, stop=settings.max_T, num=nb_ticks), decimals=1)
    new_xticks_bandwidth = np.linspace(0, settings.number_points, nb_ticks)
    new_yticks_bandwidth = np.linspace(0, settings.number_points, nb_ticks)

    cbar = plt.colorbar(contour_plot, ax=ax_bandwidth)
    cbar.outline.set_visible(False)

    ax_bandwidth.set_xlabel('Cavity length L (m)')
    ax_bandwidth.set_ylabel('Transmission coefficient')
    cbar.set_label(r'Bandwidth $\Delta$ (MHz) in log scale')

    for central_freq, colour in zip(bandwidth_list, bandwidth_colours):

        # Find couple (L, T) such that Delta within range
        boundary_down = (central_freq - settings.range_freq * central_freq) * 1e-6
        boundary_up = (central_freq + settings.range_freq * central_freq) * 1e-6
        # print(boundary_up)
        # print(boundary_down)
        indices = np.where((bandwidth_meshgrid < boundary_up) & (
                    bandwidth_meshgrid > boundary_down))  # here np.where gives a tuple, the first element of which gives the row index, while the

        # gives row index, 2nd element gives column index
        length_range = L[indices]
        transmission_range = T[indices]

        cbar.set_ticks([])
        cbar.ax.axhline(y=central_freq*1e-6, color=colour, linewidth=2, alpha=settings.alpha)
        # line_text = cbar.ax.text(central_freq*1e-6, 1, f'{central_freq*1e-6}', ha='left', va='center')
        # cbar.ax.axhline(y=boundary_down, color=colour, linestyle='--', linewidth=2)
        # cbar.ax.axhline(y=boundary_up, color=colour, linestyle='--', linewidth=2)

        ax_bandwidth.plot(length_range, transmission_range, c=colour, alpha=settings.alpha, label=f'{central_freq*1e-6}MHz')
        ax_bandwidth.legend(labelcolor='linecolor')

    # Add a text box for the parameters
    # box_text = f"Cavity loss: {settings.cavity_loss}\nCentral frequency ($\pm$ {settings.range_freq * 100}%)"
    # text_box = AnchoredText(box_text, frameon=True, loc=4, pad=0.5)
    # plt.setp(text_box.patch, facecolor='white', alpha=0.9)
    # plt.gca().add_artist(text_box)

    plt.show()

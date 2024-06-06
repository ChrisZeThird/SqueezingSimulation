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
    bandwidth_cmaps = [mplp.symmetrical_colormap((x, None)) for x in ['Blues', 'Greens', 'Reds']]

    # Bandwidth surface plot
    clev = np.arange(bandwidth_meshgrid.min(), bandwidth_meshgrid.max(), 0.2)

    fig_bandwidth = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(3, 2, 6)
    ax2 = plt.subplot(3, 2, 4)
    ax3 = plt.subplot(3, 2, 2)
    ax_bandwidth = plt.subplot(1, 3, 1)
    axes = [ax1, ax2, ax3]

    # contour_plot = ax_bandwidth.contourf(L, T, bandwidth_meshgrid, clev, cmap=mplp.cmap, norm=matplotlib.colors.LogNorm(vmin=1, vmax=bandwidth_meshgrid.max()))
    # contour_plot.set_clim(vmin=1, vmax=round(bandwidth_meshgrid.max()))
    # cbar = plt.colorbar(contour_plot, ax=ax_bandwidth)
    # cbar.outline.set_visible(False)

    # Adjust ticks for colorbar
    nb_ticks = 7
    xticks_bandwidth = np.around(np.linspace(start=settings.min_L, stop=settings.max_L, num=nb_ticks), decimals=1)
    yticks_bandwidth = np.around(np.linspace(start=settings.min_T, stop=settings.max_T, num=nb_ticks), decimals=1)
    new_xticks_bandwidth = np.linspace(0, settings.number_points, nb_ticks)
    new_yticks_bandwidth = np.linspace(0, settings.number_points, nb_ticks)

    # Prepare the custom colours to update the cbar with
    # cbar_newcolors = []

    for central_freq, colour, selected_cmap, axis in zip(bandwidth_list, bandwidth_colours, bandwidth_cmaps, axes):

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

        # Mask the region outside the desired range
        mask = np.ones_like(bandwidth_meshgrid, dtype=bool)
        mask[indices] = False
        masked_bandwidth = np.ma.masked_array(bandwidth_meshgrid, mask)

        # cbar.ax.axhline(y=central_freq*1e-6, color=colour, linewidth=2, alpha=settings.alpha)
        # line_text = cbar.ax.text(central_freq*1e-6, 1, f'{central_freq*1e-6}', ha='left', va='center')
        # cbar.ax.axhline(y=boundary_down, color=colour, linestyle='--', linewidth=2)
        # cbar.ax.axhline(y=boundary_up, color=colour, linestyle='--', linewidth=2)

        # Plot the masked region with a different colormap
        contour_zone = ax_bandwidth.contourf(L, T, masked_bandwidth, cmap=selected_cmap, alpha=settings.alpha)
        axis.axis('off')
        plt.colorbar(contour_zone, ax=axis)

        # Plot the boundaries for the region
        # ax_bandwidth.plot(L[indices], T[indices], c=colour, alpha=settings.alpha, label=f'{central_freq * 1e-6}MHz')

        # ax_bandwidth.plot(length_range, transmission_range, c=colour, alpha=settings.alpha, label=f'{central_freq*1e-6}MHz')
        # ax_bandwidth.legend(labelcolor='linecolor')

    ax_bandwidth.set_xlabel('Cavity length L (m)')
    ax_bandwidth.set_ylabel('Transmission coefficient')
    # cbar.set_label(r'Bandwidth $\Delta$ (MHz) in log scale')
    # cbar.set_ticks([])
    plt.tight_layout()
    plt.show()

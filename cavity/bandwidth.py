from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

from utils.settings import settings
import utils.plot_parameters as mplp

import cavity.cavity_formulas as cf


def bandwidth(bandwidth_list=np.array([6, 10, 20])*1e6):
    # Setting parameters
    cavity_lengths = np.linspace(start=settings.min_L, stop=settings.max_L, num=settings.number_points)
    transmission_coefficients = np.linspace(start=settings.min_T, stop=settings.max_T, num=settings.number_points)

    L, T = np.meshgrid(cavity_lengths, transmission_coefficients)
    bandwidth_meshgrid = cf.Bandwidth_bowtie(T=T, L=L, Loss=settings.cavity_loss) * 1e-6

    # Define the colours and colormaps for the bandwidths
    bandwidth_colours = ['lightsteelblue', 'cornflowerblue', 'royalblue']
    bandwidth_cmaps = [mplp.symmetrical_colormap((x, None)) for x in ['Blues', 'Greens', 'Reds']]

    # Bandwidth surface plot
    clev = np.arange(bandwidth_meshgrid.min(), bandwidth_meshgrid.max(), 0.2)

    fig_bandwidth = plt.figure(figsize=(16, 9))
    # Creating three axes: add_axes([xmin,ymin,dx,dy])
    ax1 = fig_bandwidth.add_axes((0.75, 0.1, 0.1, 0.28))
    ax2 = fig_bandwidth.add_axes((0.75, 0.395, 0.1, 0.28))
    ax3 = fig_bandwidth.add_axes((0.75, 0.69, 0.1, 0.28))
    ax_bandwidth = fig_bandwidth.add_axes((0.1, 0.1, 0.7, 0.87))
    axes = [ax1, ax2, ax3]

    for central_freq, colour, selected_cmap, axis in zip(bandwidth_list, bandwidth_colours, bandwidth_cmaps, axes):

        # Find couple (L, T) such that Delta within range
        boundary_down = (central_freq - settings.range_freq * central_freq) * 1e-6
        boundary_up = (central_freq + settings.range_freq * central_freq) * 1e-6
        indices = np.where((bandwidth_meshgrid < boundary_up) & (
                    bandwidth_meshgrid > boundary_down))  # here np.where gives a tuple, the first element of which gives the row index, while the

        # Mask the region outside the desired range
        mask = np.ones_like(bandwidth_meshgrid, dtype=bool)
        mask[indices] = False
        masked_bandwidth = np.ma.masked_array(bandwidth_meshgrid, mask)

        # Plot the masked region with a different colormap
        contour_zone = ax_bandwidth.contourf(L, T, masked_bandwidth, cmap=selected_cmap, alpha=settings.alpha)
        axis.axis('off')
        cbar_zone = plt.colorbar(contour_zone, ax=axis)
        cbar_zone_ticks = cbar_zone.get_ticks()
        cbar_zone.set_ticks(cbar_zone_ticks[1:-1])

    # Add some points to give couples from different articles
    values = {"Masada": (500e-3, 0.1), "Burks": (550e-3, 0.07), "Tanimura": (600e-3, 0.1), "Aoki": (215e-3, 0.15), "Hétet": (600e-3, 0.18)}

    # Plot all in black with star markers and a unique index label
    for i, (key, (x, y)) in enumerate(values.items(), start=1):
        ax_bandwidth.scatter(x, y, s=90, marker="*", color="black", label=f"{key}")
        ax_bandwidth.text(x + 5e-3, y + 0.002, f"{i}", color="black", fontsize=9, ha='left', va='bottom')

    # Set axis labels
    ax_bandwidth.set_xlabel('Cavity length L (m)')
    ax_bandwidth.set_ylabel('Transmission coefficient')

    # Legend with text-only entries (no markers)
    legend_elements = [
        mpatches.Patch(color='none', label=f"{i}. {key}")  # transparent patch (invisible)
        for i, key in enumerate(values.keys(), start=1)
    ]

    ax_bandwidth.legend(handles=legend_elements, loc="upper left", prop={'size': 15}, handlelength=0, handletextpad=0)
    # Add text vertically on the right side of the axes
    text_y_pos = 0.5  # Vertically centered position
    text_x_pos = 0.92  # Right side position
    text_vertical = "Bandwidth Δ (MHz)"
    plt.text(text_x_pos, text_y_pos, text_vertical, horizontalalignment='center', verticalalignment='center',
             rotation=270, transform=fig_bandwidth.transFigure, fontsize=mplp.MEDIUM_SIZE)

    plt.show()


if __name__ == "__main__":
    bandwidth()

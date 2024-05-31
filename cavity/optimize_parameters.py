import matplotlib
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np

import cavity.cavity_formulas as cf
import cavity.finding_distances as fd
from utils.settings import settings
import utils.plot_parameters as mplp

# -- SETTING PARAMETERS -- #
plot_bandwidth = settings.plot_bandwidth
plot_waist = settings.plot_waist
print(plot_waist)
# Some constant
c = settings.c
number_points = settings.number_points

# Cavity length
min_L = settings.min_L
max_L = settings.max_L
cavity_lengths = np.linspace(start=min_L, stop=max_L, num=number_points)

# Transmission coefficient
min_T = settings.min_T
max_T = settings.max_T
transmission_coefficients = np.linspace(start=min_T, stop=max_T, num=number_points, endpoint=False)  # avoid division by 0 in FSR/F

L, T = np.meshgrid(cavity_lengths, transmission_coefficients)

# Some fixed cavity parameters
d_curved = np.linspace(start=0, stop=100, num=number_points) * 1e-3

R = settings.R  # Radii of curvature

crystal_length = settings.crystal_length
index_PPKTP = settings.crystal_index  # refractive index
cavity_loss = settings.cavity_loss  # set low loss

wavelength = settings.wavelength

fixed_length = settings.fixed_length

# -- OPTIMIZE BANDWIDTH -- #
bandwidth_meshgrid = cf.Bandwidth_bowtie(T=T, L=L, Loss=cavity_loss) * 1e-6

# Find couple (L, T) such that Delta within range
central_freq = settings.central_freq  # MHz
threshold = settings.range_freq
boundary_down = central_freq - threshold * central_freq
boundary_up = central_freq + threshold * central_freq

indices = np.where((bandwidth_meshgrid < 11) & (bandwidth_meshgrid > 9))  # np.where gives tuple, 1rst element of which
                                                                        # gives row index, 2nd element gives column index
length_range = L[indices]
transmission_range = T[indices]

# Plot bandwidth
if plot_bandwidth:
    clev = np.arange(bandwidth_meshgrid.min(), bandwidth_meshgrid.max(), 0.2)

    fig_bandwidth, ax_bandwidth = plt.subplots()
    contour_plot = ax_bandwidth.contourf(L, T, bandwidth_meshgrid, clev, cmap=mplp.cmap)

    # Adjust ticks for colorbar
    nb_ticks = 7
    xticks_bandwidth = np.around(np.linspace(start=min_L, stop=max_L, num=nb_ticks), decimals=1)
    yticks_bandwidth = np.around(np.linspace(start=min_T, stop=max_T, num=nb_ticks), decimals=1)
    new_xticks_bandwidth = np.linspace(0, number_points, nb_ticks)
    new_yticks_bandwidth = np.linspace(0, number_points, nb_ticks)

    cbar = plt.colorbar(contour_plot, ax=ax_bandwidth)
    cbar.outline.set_visible(False)

    cbar.ax.axhline(y=central_freq, color='white', linewidth=50, alpha=settings.alpha)
    ax_bandwidth.plot(length_range, transmission_range, c='white', alpha=settings.alpha)

    ax_bandwidth.set_xlabel('Cavity length L (m)')
    ax_bandwidth.set_ylabel('Transmission coefficient')
    ax_bandwidth.set_title(f'Intra-cavity loss: L = {cavity_loss}')
    cbar.set_label(r'Bandwidth $\Delta$ (MHz)')

    plt.show()

# -- OPTIMIZE LENGTHS -- #
w1, w2, valid_indices_1, valid_indices_2 = cf.Beam_waist(d_curved=d_curved,
                                                       L=fixed_length,
                                                       R=R,
                                                       l_crystal=crystal_length,
                                                       index_crystal=index_PPKTP,
                                                       wavelength=wavelength)


if plot_waist:
    fig_waist, ax1 = plt.subplots(figsize=(10, 10))

    # Add waist 1
    color1 = 'tab:red'
    ax1.set_xlabel(r'Distance $d_{c}$ (m)')
    ax1.set_ylabel(r'Beam waist size $w_1$ (mm)', color=color1)
    ax1.plot(d_curved[valid_indices_1], w1[valid_indices_1] * 1e3, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Annotate maximum value
    max_index = np.argmax(w1[valid_indices_1])
    min_index = np.argmin(w1[valid_indices_1])
    x_max = d_curved[valid_indices_1][max_index]
    y_max = w1[valid_indices_1][max_index] * 1e3
    y_min = w1[valid_indices_1][min_index] * 1e3
    ax1.vlines(x=x_max, ymin=y_min, ymax=y_max, color='r', linestyles='--')
    ax1.hlines(y=y_max, xmin=d_curved[valid_indices_1][0], xmax=x_max, color='r', linestyles='--')
    ax1.plot(x_max, y_max, color='r', marker='o')
    ax1.text(x_max, y_max + 0.0005, r'$(d_{c}, w_1) = $' + f'({x_max:.3f}, {y_max:.3f})', color='r')

    # Add waist 2
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color2 = 'tab:blue'
    ax2.set_ylabel(r'Beam waist size $w_2$ (mm)', color=color2)
    ax2.plot(d_curved[valid_indices_2], w2[valid_indices_2] * 1e3, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add a text box for the parameters
    box_text = f"Cavity length: {fixed_length * 1e3} mm\nMirror radius: {R * 1e3} mm\nCrystal length: {crystal_length * 1e3}mm"
    text_box = AnchoredText(box_text, frameon=True, loc=4, pad=0.5)
    plt.setp(text_box.patch, facecolor='white', alpha=settings.alpha)
    plt.gca().add_artist(text_box)

    fig_waist.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    rayleigh_length = index_PPKTP * np.pi * (y_max * 1e-3) / wavelength
    print(rayleigh_length)


# -- Kaertner class notes -- #
plot_kaertner = settings.plot_kaertner

if plot_kaertner:
    R1 = settings.R1
    R2 = settings.R2
    length_kaertner = np.linspace(start=0, stop=20, num=number_points) * 1e-2

    w1 = cf.waist_mirror1(R1=R1, R2=R2, L=length_kaertner, wavelength=790e-9)
    w2 = cf.waist_mirror3(R1=R1, R2=R2, L=length_kaertner, wavelength=790e-9)
    w0 = cf.waist_intracavity(R1=R1, R2=R2, L=length_kaertner, wavelength=790e-9)

    fig_waist_kaertner, ax_kaertner = plt.subplots(nrows=3, ncols=1)

    ax_kaertner[0].plot(length_kaertner * 1e2, w1 / np.sqrt(wavelength * R1 / np.pi), color='red')
    ax_kaertner[0].set_ylabel(r'$w_1 / (\lambda R_1 / \pi)^{1/2}$', fontsize=12)

    ax_kaertner[1].plot(length_kaertner * 1e2, w2 / np.sqrt(wavelength * R2 / np.pi), color='red')
    ax_kaertner[1].set_ylabel(r'$w_2 / (\lambda R_2 / \pi)^{1/2}$', fontsize=12)

    ax_kaertner[2].plot(length_kaertner * 1e2, w0 / np.sqrt(wavelength / np.pi), color='red')
    ax_kaertner[2].set_ylabel(r'$w_0 / (\lambda / \pi)^{1/2}$', fontsize=12)

    ax_kaertner[2].set_xlabel('Cavity length L (cm)', fontsize=18)

    fig_waist_kaertner.tight_layout()
    plt.show()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import cavity.cavity_formulas as cf
import cavity.finding_distances as fd
from utils.settings import settings
import utils.plot_parameters as mplp

# -- SETTING PARAMETERS -- #
plot_bandwidth = False
plot_waist = False

# Some constant
c = settings.c
number_points = 500

# Cavity length
min_L = 0.25
max_L = 2.0
cavity_lengths = np.linspace(start=min_L, stop=max_L, num=number_points)

# Transmission coefficient
min_T = 0.01
max_T = 0.5
transmission_coefficients = np.linspace(start=min_T, stop=max_T, num=number_points, endpoint=False)  # avoid division by 0 in FSR/F

L, T = np.meshgrid(cavity_lengths, transmission_coefficients)

# Some fixed cavity parameters
d_curved = np.linspace(start=0, stop=100, num=number_points) * 1e-3
cavity_width = np.linspace(start=0.010, stop=0.020, num=10)

R = 50e-3  # Radii of curvature

crystal_length = 10e-3

cavity_loss = 0.004  # set low loss

# -- OPTIMIZE BANDWIDTH -- #
bandwidth_meshgrid = cf.Bandwidth(T=T, L=L, Loss=cavity_loss) * 1e-6

# Find couple (L, T) such that Delta within range
central_freq = 6  # MHz
threshold = 0.1
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

    cbar.ax.axhline(y=central_freq, color='white', linewidth=50, alpha=mplp.alpha)
    ax_bandwidth.plot(length_range, transmission_range, c='white', alpha=mplp.alpha)

    ax_bandwidth.set_xlabel('Cavity length L (m)')
    ax_bandwidth.set_ylabel('Transmission coefficient')
    ax_bandwidth.set_title(f'Intra-cavity loss: L = {cavity_loss}')
    cbar.set_label(r'Bandwidth $\Delta$ (MHz)')

    plt.show()

# -- OPTIMIZE LENGTHS -- #
wavelength = 860e-9
index_PPKTP = 1.8396  # refractive index
fixed_length = 500e-3
R = 50e-3

# w1, w2, valid_indices = cf.Beam_waist(d_curved=d_curved,
#                        L=500e-3,
#                        cavity_width=15e-3,
#                        R=R,
#                        l_crystal=crystal_length,
#                        index_crystal=index_PPKTP,
#                        wavelength=wavelength,
#                        tamagawa=True)

d_flat, OF, OC, _, _ = fd.finding_unknown_distance(L=fixed_length, R=R, l=crystal_length, d_curved=d_curved)
A, B, C, D = cf.ABCD_Matrix(d_curved=d_curved, d_flat=d_flat, d_diag=OF + OC, R=R, l_crystal=crystal_length, index_crystal=index_PPKTP)
waist, index = cf.waist_from_q(A=A, B=B, C=C, D=D, wavelength=wavelength, crystal_index=index_PPKTP)

# print(d_curved[:10], d_flat[:10])  # checking d_curved < d_flat

if plot_waist:
    fig_waist, ax1 = plt.subplots()

    # print(d_curved[valid_indices].shape)

    ax1.set_xlabel(r'Distance $d_{c}$ (m)')
    ax1.set_ylabel(r'Beam waist size $w_0$ (mm)')
    # print(d_curved[index].shape, waist.shape)
    ax1.plot(d_curved[index], waist * 1e3)
    # ax1.plot(d_curved, waist * 1e3)
    ax1.tick_params(axis='y')

    # color1 = 'tab:red'
    # ax1.set_xlabel(r'Distance $d_{c}$ (m)')
    # ax1.set_ylabel(r'Beam waist size $w_1$ (mm)', color=color1)
    # ax1.plot(d_curved[valid_indices], w1 * 1e3, color=color1)
    # ax1.tick_params(axis='y', labelcolor=color1)
    #
    # ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    # color2 = 'tab:blue'
    # ax2.set_ylabel(r'Beam waist size $w_2$ (mm)', color=color2)
    # ax2.plot(d_curved[valid_indices], w2 * 1e3, color=color2)
    # ax2.tick_params(axis='y', labelcolor=color2)

    fig_waist.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

kaertner = True

if kaertner:
    R1 = 10e-2
    R2 = 11e-2
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

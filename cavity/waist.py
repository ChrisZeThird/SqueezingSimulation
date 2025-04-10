from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

import numpy as np

import cavity.cavity_formulas as cf
from utils.settings import settings
import utils.plot_parameters


def waist():
    """
    Plot the waist with respect to the requested parameters, by default 'dc'. Other parameters are 'L', 'R', 'lc' (length
    crystal)
    :return: A plot of the waist
    """
    # Default values
    kwargs = {
        'd_curved': settings.fixed_d_curved,
        'L': settings.fixed_length,
        'R': settings.R,
        'l_crystal': settings.crystal_length,
        'index_crystal': settings.crystal_index,
        'wavelength': settings.wavelength
    }

    plot_vs = settings.waist_vs  # settings to plot waist against specific parameter

    # Determine the sweep variable and generate the array accordingly
    if plot_vs == 'dc':
        sweep_array = np.linspace(start=settings.d_curved_min, stop=settings.d_curved_max,
                                  num=settings.number_points)
        kwargs['d_curved'] = sweep_array
        xlabel = r'Distance $d_{c}$ (mm)'

        box_text = (f"Fixed values:\n"
                    f"L = {settings.fixed_length * 1e3:.1f} mm\n"
                    f"R = {settings.R * 1e3:.1f} mm\n"
                    f"$l_c$ = {settings.crystal_length * 1e3:.1f} mm\n"
                    f"$\lambda$ = {settings.wavelength * 1e9} nm")

    elif plot_vs == 'L':
        sweep_array = np.linspace(start=settings.min_L, stop=settings.max_L, num=settings.number_points)
        kwargs['L'] = sweep_array
        xlabel = r'Cavity round-trip length $L$ (m)'

        box_text = (f"Fixed values:\n"
                    f"$d_c$ = {settings.fixed_d_curved * 1e3:.1f} mm\n"
                    f"R = {settings.R * 1e3:.1f} mm\n"
                    f"$l_c$ = {settings.crystal_length * 1e3:.1f} mm\n"
                    f"$\lambda$ = {settings.wavelength * 1e9} nm")

    elif plot_vs == 'R':
        sweep_array = np.linspace(start=settings.min_R, stop=settings.max_R, num=settings.number_points)
        kwargs['R'] = sweep_array
        xlabel = r'Mirror curvature radius $R$ (mm)'

        box_text = (f"Fixed values:\n"
                    f"$d_c$ = {settings.fixed_d_curved * 1e3:.1f} mm\n"
                    f"L = {settings.fixed_length * 1e3:.1f} mm\n"
                    f"$l_c$ = {settings.crystal_length * 1e3:.1f} mm\n"
                    f"$\lambda$ = {settings.wavelength * 1e9} nm")

    elif plot_vs == 'lc':
        sweep_array = np.linspace(start=settings.min_lc, stop=settings.max_lc, num=settings.number_points)
        kwargs['l_crystal'] = sweep_array
        xlabel = r'Crystal length $l_c$ (mm)'

        box_text = (f"Fixed values:\n"
                    f"$d_c$ = {settings.fixed_d_curved * 1e3:.1f} mm\n"
                    f"L = {settings.fixed_length * 1e3:.1f} mm\n"
                    f"R = {settings.R * 1e3:.1f} mm\n"
                    f"$\lambda$ = {settings.wavelength * 1e9} nm")

    else:
        raise ValueError(f"Invalid plot_vs value '{plot_vs}'. Choose from 'dc', 'L', 'R', or 'lc'.")

    # print(kwargs)
    z1, z2, w1, w2, valid_indices = cf.Beam_waist(**kwargs)

    if valid_indices[0][0].size == 0 or valid_indices[1][0].size == 0:
        print("Invalid values encountered, can't proceed.")
        print('q1: ', z1)
        print('q2: ', z2)
        return  # or return some default value / raise an exception

    # Plot waist
    fig_waist, ax1 = plt.subplots(figsize=(16, 9))

    color1 = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(r'Beam waist size $w_1$ (mm)', color=color1)
    ax1.plot(sweep_array[valid_indices[0]] * 1e3, w1[valid_indices[0]] * 1e3, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    color2 = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Beam waist size $w_2$ (mm)', color=color2)
    ax2.plot(sweep_array[valid_indices[1]] * 1e3, w2[valid_indices[1]] * 1e3, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Display parameters used
    text_box = AnchoredText(box_text, frameon=True, loc='upper right', pad=0.5)
    plt.setp(text_box.patch, facecolor='white', alpha=settings.alpha)
    plt.gca().add_artist(text_box)

    fig_waist.tight_layout()
    plt.show()


def Kaertner():
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

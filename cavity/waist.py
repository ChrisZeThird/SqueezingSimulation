import matplotlib
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt

import numpy as np

import cavity.cavity_formulas as cf
from utils.settings import settings
import utils.plot_parameters


def waist():
    d_curved = np.linspace(start=settings.d_curved_min, stop=settings.d_curved_max, num=settings.number_points)

    w1, w2, valid_indices_1, valid_indices_2 = cf.Beam_waist(d_curved=d_curved,
                                                             L=settings.fixed_length,
                                                             R=settings.R,
                                                             l_crystal=settings.crystal_length,
                                                             index_crystal=settings.crystal_index,
                                                             wavelength=settings.wavelength)

    # Plot waist
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
    y_max = w1[valid_indices_1][max_index]
    y_min = w1[valid_indices_1][min_index]
    ax1.vlines(x=x_max, ymin=y_min*1e3, ymax=y_max*1e3, color='r', linestyles='--')
    ax1.hlines(y=y_max*1e3, xmin=d_curved[valid_indices_1][0], xmax=x_max, color='r', linestyles='--')
    ax1.plot(x_max, y_max*1e3, color='r', marker='o')
    ax1.text(x_max, y_max*1e3 + 0.0003, r'$(d_{c}, w_1) = $' + f'({x_max:.3f}, {(y_max * 1e3):.3f})', color='r')

    # Add waist 2
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color2 = 'tab:blue'
    ax2.set_ylabel(r'Beam waist size $w_2$ (mm)', color=color2)
    ax2.plot(d_curved[valid_indices_2], w2[valid_indices_2] * 1e3, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Print Rayleigh length
    rayleigh_length = settings.crystal_index * np.pi * (y_max ** 2) / settings.wavelength
    print(rayleigh_length)

    # Add a text box for the parameters
    box_text = f"Cavity length: {settings.fixed_length * 1e3} mm\nMirror radius: {settings.R * 1e3} mm\nCrystal length: {settings.crystal_length * 1e3}mm"
    text_box = AnchoredText(box_text, frameon=True, loc=4, pad=0.5)
    plt.setp(text_box.patch, facecolor='white', alpha=settings.alpha)
    plt.gca().add_artist(text_box)

    fig_waist.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

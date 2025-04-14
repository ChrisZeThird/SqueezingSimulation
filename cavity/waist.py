from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import cavity.cavity_formulas as cf
from utils.settings import settings
import utils.plot_parameters

import csv
import os
from datetime import datetime


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
    # plt.show()

    # Find max of w1
    max_idx = np.argmax(w1[valid_indices[0]])
    max_w1 = w1[valid_indices[0]][max_idx]
    associated_w2 = w2[valid_indices[0]][max_idx]
    associated_param = sweep_array[valid_indices[0]][max_idx]

    row = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'sweep_param': plot_vs,
        'sweep_value_mm': associated_param * 1e3,
        'max_w1_mm': max_w1 * 1e3,
        'associated_w2_mm': associated_w2 * 1e3,
        'L_mm': settings.fixed_length * 1e3,
        'd_curved_mm': settings.fixed_d_curved * 1e3,
        'R_mm': settings.R * 1e3,
        'l_crystal_mm': settings.crystal_length * 1e3,
        'index_crystal': settings.crystal_index,
        'wavelength_nm': settings.wavelength * 1e9
    }

    # Define file path
    log_path = 'waist_log.csv'
    file_exists = os.path.isfile(log_path)

    # Append to CSV
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def plot_from_csv(filename="waist_log.csv"):
    # Load the CSV
    df = pd.read_csv(filename)

    # Filter only for given sweep
    df_dc = df[df["sweep_param"] == settings.waist_vs]

    # Group by wavelength and plot one line per group
    plt.figure(figsize=(8, 5))

    for wavelength, group in df_dc.groupby("wavelength_nm"):
        group_sorted = group.sort_values("L_mm")
        plt.plot(group_sorted["L_mm"],
                 group_sorted["sweep_value_mm"],
                 marker='o',
                 label=f"{wavelength} nm",
                 alpha=1)
        break

    plt.xlabel("$L$ (mm)")
    plt.ylabel("$d_c$ (mm) ")
    plt.title("Optimal Mirror Distance $d_c$ vs Cavity Length $L$ for max $w_1$")
    # plt.legend(title="Wavelength")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

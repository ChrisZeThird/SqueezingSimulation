from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

import cavity.cavity_formulas as cf
from utils.settings import settings
import utils.plot_parameters as pm
from utils.settings import settings

import csv
import os
from datetime import datetime


def waist():
    """
    Plot the waist with respect to the requested parameters, by default 'dc'. Other parameters are 'L', 'R', 'lc' (length
    crystal)
    :return: A plot of the waist
    """
    # length_test = np.linspace(start=500, stop=800, num=100) * 1e-3
    length_test = np.array([settings.fixed_length])
    for i in range(len(length_test)):
        # Default values
        kwargs = {
            'd_curved': settings.fixed_d_curved,
            'L': length_test[i],
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
                        f"L = {length_test[i] * 1e3:.1f} mm\n"
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

        print(r"Valid range of $d_c$ values: ", sweep_array[valid_indices[0]][0] * 1e3,  sweep_array[valid_indices[0]][-1] * 1e3)

        x_valid = sweep_array[valid_indices[0]]
        w1_valid = w1[valid_indices[0]]
        w2_valid = w2[valid_indices[1]]

        # Plot waist
        fig_waist, ax1 = plt.subplots(figsize=(16, 9))

        color1 = 'tab:red'
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(r'Beam waist size $w_1$ (mm)', color=color1)
        ax1.plot(x_valid * 1e3, w1_valid * 1e3, color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        color2 = 'tab:blue'
        ax2 = ax1.twinx()
        ax2.set_ylabel(r'Beam waist size $w_2$ (mm)', color=color2)
        ax2.plot(x_valid * 1e3,  w2_valid * 1e3, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        # Display parameters used
        text_box = AnchoredText(box_text, frameon=True, loc='upper right', pad=0.5)
        plt.setp(text_box.patch, facecolor='white', alpha=settings.alpha)
        plt.gca().add_artist(text_box)

        # Find max of w1, corresponding w2 and d_c
        max_index = np.argmax(w1_valid)

        # Add vertical line at the selected d_c
        selected_dc = x_valid[max_index]  # in mm
        ax1.axvline(selected_dc * 1e3, color='gray', linestyle='--', linewidth=1)
        ax1.annotate(f"{selected_dc * 1e3:.1f} mm", xy=(selected_dc * 1e3, np.min(w1_valid) * 1e3),
                     xytext=(selected_dc * 1e3 + + 0.1, np.min(w1_valid) * 1e3 - 0.0015),
                     fontsize=20, color='gray')

        # Add horizontal line for w1 at that d_c
        selected_w1 = w1_valid[max_index]  # in mm
        ax1.axhline(selected_w1 * 1e3, color='tab:red', linestyle='--', linewidth=1)
        ax1.annotate(f"{selected_w1 * 1e3:.3f} mm", xy=(selected_dc * 1e3, selected_w1 * 1e3),
                     xytext=(selected_dc * 1e3 + 0.1, selected_w1 * 1e3 + 0.0005),
                     fontsize=20, color='tab:red')

        # Add horizontal line for w2 at that d_c
        selected_w2 = w2_valid[max_index]  # in mm
        ax2.axhline(selected_w2 * 1e3, color='tab:blue', linestyle='--', linewidth=1)
        ax2.annotate(f"{selected_w2 * 1e3:.3f} mm", xy=(selected_dc * 1e3, selected_w2 * 1e3),
                     xytext=(selected_dc * 1e3 + 0.1, selected_w2 * 1e3 + 0.005),
                     fontsize=20, color='tab:blue')

        fig_waist.tight_layout()
        plt.show()

        # Find max of w1
        # max_idx = np.argmax(w1[valid_indices[0]])
        # max_w1 = w1[valid_indices[0]][max_idx]
        # associated_w2 = w2[valid_indices[0]][max_idx]
        # associated_param = sweep_array[valid_indices[0]][max_idx]
        #
        # row = {
        #     'timestamp': datetime.now().isoformat(timespec='seconds'),
        #     'sweep_param': plot_vs,
        #     'sweep_value_mm': associated_param * 1e3,
        #     'max_w1_mm': max_w1 * 1e3,
        #     'associated_w2_mm': associated_w2 * 1e3,
        #     'L_mm': length_test[i] * 1e3,  # length_test[i]
        #     'd_curved_mm': settings.fixed_d_curved * 1e3,
        #     'R_mm': settings.R * 1e3,
        #     'l_crystal_mm': settings.crystal_length * 1e3,
        #     'index_crystal': settings.crystal_index,
        #     'wavelength_nm': settings.wavelength * 1e9
        # }
        #
        # # Define file path
        # log_path = 'waist_log.csv'
        # file_exists = os.path.isfile(log_path)
        #
        # # Append to CSV
        # with open(log_path, 'a', newline='') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        #     if not file_exists:
        #         writer.writeheader()
        #     writer.writerow(row)


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


def plot_w1_w2_vs_L(filename="waist_log.csv", wavelength_nm=780.0):
    # Load the CSV
    df = pd.read_csv(filename)

    # Filter only for the fixed wavelength and sweep type
    df_filtered = df[
        (df["wavelength_nm"] == wavelength_nm) &
        (df["sweep_param"] == settings.waist_vs)
    ]

    # Get unique R values and assign colors
    unique_Rs = sorted(df_filtered["R_mm"].unique())
    colors = cm.viridis(np.linspace(0, 1, len(unique_Rs)))

    fig, ax = plt.subplots(figsize=(18, 9))

    for color, R in zip(colors, unique_Rs):
        df_R = df_filtered[df_filtered["R_mm"] == R].sort_values("L_mm")

        # Plot w1 and w2 with same color
        ax.plot(df_R["L_mm"], df_R["max_w1_mm"] * 1e3, color=color, linestyle='-')  # w1
        ax.plot(df_R["L_mm"], df_R["associated_w2_mm"] * 1e3, color=color, linestyle='--')  # w2

    # Legend for w1 and w2
    legend_waists = ax.legend(
        handles=[
            Line2D([0], [0], color='black', linestyle='-', label="$w_1$"),
            Line2D([0], [0], color='black', linestyle='--', label="$w_2$")
        ],
        loc='upper left',
        bbox_to_anchor=(1.02, 1.02),
        title='Waists',
        title_fontsize=pm.MEDIUM_SIZE-5
    )

    legend_waists._legend_box.align = "left"
    ax.add_artist(legend_waists)

    # Legend for R values
    legend_R = ax.legend(
        handles=[
            Line2D([0], [0], color=color, linestyle='-', linewidth=2, label=f"${R:.3f}$ mm")
            for color, R in zip(colors, unique_Rs)
        ],
        loc='lower left',
        bbox_to_anchor=(1.02, 0.0),
        title="$R$ (Mirror Radius)",
        title_fontsize=pm.MEDIUM_SIZE-5,
        fontsize=18
    )
    legend_R._legend_box.align = "left"
    ax.add_artist(legend_R)

    # Labels and layout
    ax.set_xlabel("$L$ (mm)")
    ax.set_ylabel("Waist size (μm)")
    ax.set_title(f"Max $w_1$ and Associated $w_2$ vs $L$ at {wavelength_nm} nm")
    ax.grid(True)

    # Increase margin for legends
    plt.subplots_adjust(right=0.5)
    plt.tight_layout()
    plt.show()


def angle_evolution(L_values, R=settings.R, l_crystal=settings.crystal_length,
                    wavelength=settings.wavelength, theta_max=15,
                    plot=False, fixed_theta_deg=None, dc_range=None):
    """
    Computes d_flat for given L, dc_range, and optional fixed theta values.

    - If fixed_theta_deg is set (single or list), plots d_flat vs dc for each theta on the same figure.
    - Else, finds dc that maximizes beam waist and plots d_flat vs theta.

    :param L_values: List of cavity lengths [m]
    :param R: Radius of curvature [m]
    :param l_crystal: Crystal length [m]
    :param wavelength: Laser wavelength [m]
    :param theta_max: Max theta to scan if sweeping [deg]
    :param plot: Enable plotting
    :param fixed_theta_deg: Single value or list of fixed angles [deg] (for d_flat vs dc plots)
    :param dc_range: Range of curved mirror distances [m]
    :return: Dict {L: (dc_opt, d_flat_array)} or None if fixed_theta mode
    """
    results = {}

    # Default dc range if not supplied
    if dc_range is None:
        dc_range = np.linspace(settings.d_curved_min, settings.d_curved_max, settings.number_points)
    else:
        dc_range = np.asarray(dc_range)

    # If fixed theta(s) are provided: plot d_flat vs dc for each theta on same figure
    if fixed_theta_deg is not None:
        if not hasattr(fixed_theta_deg, '__iter__'):
            fixed_theta_deg = [fixed_theta_deg]

        for L in L_values:
            plt.figure()
            for theta_deg in fixed_theta_deg:
                theta_rad = np.deg2rad(theta_deg)
                x = np.cos(theta_rad) / (1 + np.cos(theta_rad))
                d_flat = L * x - dc_range

                plt.plot(dc_range * 1e3, d_flat * 1e3,
                         label=rf"$\theta = {theta_deg}^\circ$")

            plt.title(rf"$d_{{\mathrm{{flat}}}}$ vs $d_c$ for $L = {L * 1e3:.0f}$ mm")
            plt.xlabel("Curved mirror distance $d_c$ [mm]")
            plt.ylabel("Flat mirror distance $d_{\\mathrm{flat}}$ [mm]")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return None

    # Else: default behavior sweeping over theta to find optimal dc
    theta_dense = np.linspace(0, theta_max, 100)
    theta_integers = np.arange(0, theta_max + 1)
    theta = np.union1d(theta_dense, theta_integers)
    theta_rad = np.deg2rad(theta)

    for L in L_values:
        z1, z2, w1, w2, valid = cf.Beam_waist(dc_range, L, R, l_crystal, wavelength=wavelength)
        valid_z1 = valid[0]
        w1_valid = w1[valid_z1]
        dc_valid = dc_range[valid_z1]

        if len(w1_valid) == 0:
            print(f"[!] No valid waist found for L = {L * 1e3:.1f} mm.")
            continue

        idx_max = np.argmax(w1_valid)
        dc_opt = dc_valid[idx_max]

        x = np.cos(theta_rad) / (1 + np.cos(theta_rad))
        d_flat_array = L * x - dc_opt
        results[L] = (dc_opt, d_flat_array)

        if plot:
            plt.plot(theta, d_flat_array * 1e3,
                     label=rf"$L={{{L * 1e3:.0f}}}$ mm; $d_c={{{dc_opt * 1e3:.2f}}}$ mm")

    if plot:
        plt.xlabel("Folding angle θ [deg]")
        plt.ylabel("Flat mirror distance $d_{\\mathrm{flat}}$ [mm]")
        plt.title("Flat mirror distance vs folding angle")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results


def plot_max_waist_vs_all():
    def compute_and_plot(ax1, ax2, param, info, waist_index=2):
        sweep_array = info['sweep']
        unit_scale = info['unit_scale']
        label = info['label']

        max_waists = []
        optimal_dc = []

        for val in sweep_array:
            d_curved_array = np.linspace(settings.d_curved_min, settings.d_curved_max, 200)

            kwargs = {
                'L': settings.fixed_length,
                'R': settings.R,
                'l_crystal': settings.crystal_length,
                'd_curved': d_curved_array,
                'index_crystal': settings.crystal_index,
                'wavelength': settings.wavelength,
            }

            if param == 'L':
                kwargs['L'] = val
            elif param == 'lc':
                kwargs['l_crystal'] = val
            elif param == 'R':
                kwargs['R'] = val

            _, _, w1, w2, (valid_z1, _) = cf.Beam_waist(**kwargs)
            valid_dc = d_curved_array[valid_z1]
            selected_waist = w1 if waist_index == 1 else w2
            valid_waist = selected_waist[valid_z1]

            if valid_waist.size > 0:
                idx_max = np.argmax(valid_waist)
                max_waists.append(valid_waist[idx_max])
                optimal_dc.append(valid_dc[idx_max])
            else:
                max_waists.append(np.nan)
                optimal_dc.append(np.nan)

        # Waist plot
        ax1.plot(sweep_array * unit_scale, [w * 1e6 for w in max_waists], color='tab:red')
        ax1.set_xlabel(label)
        if i != 1:
            ax1.set_ylabel(f'Max beam waist $w_{"1" if waist_index == 1 else "2"}$ (µm)', color='tab:red')
        else:
            ax1.set_ylabel('')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        # Optimal d_curved plot
        ax2.plot(sweep_array * unit_scale, [optimal_dc_value * 1e3 for optimal_dc_value in optimal_dc], color='tab:blue', linestyle='--')
        if i != 0:
            ax2.set_ylabel('Optimal $d_c$ (mm)', color='tab:blue')
        else:
            ax2.set_ylabel('')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        ax1.set_title(f'Max Waist and Optimal $d_c$ vs {param}')
        ax1.grid(True)

        # Add fixed parameter box
        fixed_params_text = "Fixed parameters:\n"
        x, y = 0.7, 0.85
        if param == 'L':
            fixed_params_text += r"$R = {}$ mm".format(settings.R * 1e3) + "\n"
            fixed_params_text += r"$l_c = {}$ mm".format(settings.crystal_length * 1e3) + "\n"
            fixed_params_text += r"$\lambda = {}$ nm".format(settings.wavelength * 1e9)
        elif param == 'lc':
            fixed_params_text += r"$R = {}$ mm".format(settings.R * 1e3) + "\n"
            fixed_params_text += r"$L = {}$ mm".format(settings.fixed_length * 1e3) + "\n"
            fixed_params_text += r"$\lambda = {}$ nm".format(settings.wavelength * 1e9)
            y = 0.3
        elif param == 'R':
            fixed_params_text += r"$L = {}$ mm".format(settings.fixed_length * 1e3) + "\n"
            fixed_params_text += r"$l_c = {}$ mm".format(settings.crystal_length * 1e3) + "\n"
            fixed_params_text += r"$\lambda = {}$ nm".format(settings.wavelength * 1e9)
            y = 0.3

        ax1.text(x, y, fixed_params_text, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left', color='black',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))

    # Settings
    parameters = {
        'L': {
            'label': 'Cavity length $L$ (mm)',
            'sweep': np.linspace(200, 800, 300) * 1e-3,
            'unit_scale': 1e3
        },
        'lc': {
            'label': 'Crystal length $l_c$ (mm)',
            'sweep': np.linspace(10, 30, 100, endpoint=True) * 1e-3,
            'unit_scale': 1e3
        },
        'R': {
            'label': 'Mirror curvature $R$ (mm)',
            'sweep': np.linspace(50, 150, 100, endpoint=True) * 1e-3,
            'unit_scale': 1e3
        }
    }

    for waist_index in [1, 2]:
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(4, 4)
        axes = [
            fig.add_subplot(gs[:2, :2]),
            fig.add_subplot(gs[:2, 2:]),
            fig.add_subplot(gs[2:, 1:3])
        ]

        for i, (param, info) in enumerate(parameters.items()):
            ax1 = axes[i]
            ax2 = ax1.twinx()
            compute_and_plot(ax1, ax2, param, info, waist_index=waist_index)

        fig.suptitle(f'Waist Analysis', fontsize=22)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

        break

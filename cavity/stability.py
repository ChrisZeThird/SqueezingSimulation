import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
from cavity.cavity_formulas import stability_condition
from utils.settings import settings

import utils.plot_parameters


# Function to format parameter names correctly for LaTeX
def format_label(param):
    if "_" in param:
        param = param.replace("_", r"_{") + "}"
    return f"${param}$"


# Constants
L = settings.fixed_length
R = settings.R
d_curved = np.linspace(start=settings.d_curved_min, stop=settings.d_curved_max, num=settings.number_points)
l_crystal_array = np.array([10, 15, 20, 25, 30]) * 1e-3  # mm to meters


# --- FUNCTIONS ---
def generate_meshgrid(param_x, values_x, param_y, values_y):
    X_grid, Y_grid = np.meshgrid(values_x, values_y)
    return X_grid, Y_grid


def plot_stability_contour(param_x, param_y, x_range, y_range, fixed_params, discrete_R=None, discrete_l_crystal=None, discrete_L=None):
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    stability_map = np.full_like(X_grid, np.nan)

    # Vectorize the stability condition calculation
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            params = fixed_params.copy()
            params[param_x] = X_grid[i, j]
            params[param_y] = Y_grid[i, j]

            stability_values = stability_condition(
                d_curved=d_curved,
                L=params['L'],
                R=params['R'],
                l_crystal=params['l_crystal'],
                index_crystal=settings.crystal_index,
                wavelength=settings.wavelength
            )

            if stability_values[1].size > 0:
                stability_map[i, j] = np.max(stability_values[1])
            else:
                stability_map[i, j] = np.nan

    plt.figure(figsize=(8, 6))
    contour = plt.pcolormesh(X_grid, Y_grid, stability_map, shading='auto', cmap=cmc.berlin)
    plt.colorbar(contour)

    # Map discrete values to parameter names
    discrete_values = {
        'R': (discrete_R, 'R'),
        'l_crystal': (discrete_l_crystal, 'l_crystal'),
        'L': (discrete_L, 'L'),
    }

    for param_name, (values, label_prefix) in discrete_values.items():
        if values is not None:
            for val in values:
                if param_x == param_name:
                    plt.axvline(x=val, color='w', linestyle='--', linewidth=1.5, zorder=10)
                elif param_y == param_name:
                    plt.axhline(y=val, color='w', linestyle='--', linewidth=1.5, zorder=10)

    param1_label = format_label(param_x)
    param2_label = format_label(param_y)
    plt.xlabel(f'{param1_label} (m)')
    plt.ylabel(f'{param2_label} (m)')
    plt.title(f"Stability Contour: {param1_label} vs {param2_label}")

    fixed_params_text = "\n".join([f"{format_label(key)} = {val * 1e3} mm" for key, val in fixed_params.items()])
    plt.gca().text(0.85, 0.95, fixed_params_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.show()


# Define parameter ranges
L_range = np.linspace(start=settings.min_L, stop=settings.max_L, num=settings.number_points)
R_range = np.linspace(start=40, stop=160, num=settings.number_points) * 1e-3
l_crystal_range = np.linspace(start=5, stop=35, num=settings.number_points) * 1e-3

default_params = {
    'L': settings.fixed_length,
    'R': settings.R,
    'l_crystal': settings.crystal_length
}


def get_fixed_params(x_param, y_param):
    return {key: val for key, val in default_params.items() if key not in (x_param, y_param)}


# Plot combinations
plot_stability_contour('R', 'L', R_range, L_range, get_fixed_params('R', 'L'), discrete_R=np.array([50, 100, 150]) * 1e-3, discrete_L=np.array([550, 790]) * 1e-3)
plot_stability_contour('l_crystal', 'L', l_crystal_range, L_range, get_fixed_params('l_crystal', 'L'),
                      discrete_l_crystal=np.array([10, 20, 30]) * 1e-3, discrete_L=np.array([550, 790]) * 1e-3)
plot_stability_contour('R', 'l_crystal', R_range, l_crystal_range, get_fixed_params('R', 'l_crystal'),
                       discrete_R=np.array([50, 100, 150]) * 1e-3, discrete_l_crystal=np.array([10, 20, 30]) * 1e-3)

# --- SINGLE PARAMETER PLOT ---
# fig, ax = plt.subplots(figsize=(12, 6))
# line_labels = []
#
# for lc in l_crystal_array:
#     d_valid, s_valid, w_valid = stability_condition(
#         d_curved=d_curved,
#         L=L,
#         R=R,
#         l_crystal=lc,
#         index_crystal=settings.crystal_index,
#         wavelength=settings.wavelength
#     )
#
#     if d_valid.size == 0:
#         continue
#
#     max_idx = np.argmax(w_valid)
#     d_max = d_valid[max_idx] * 1e3
#     s_max = s_valid[max_idx]
#     w_max = w_valid[max_idx] * 1e6
#
#     label = f'$l_{{c}}$ = {lc*1e3:.0f} mm, $w_{{max}}$ = {w_max:.1f} µm'
#     line, = ax.plot(d_valid * 1e3, s_valid, label=label)
#     color = line.get_color()
#
#     ax.plot(d_max, s_max, 'o', color=color, markersize=6)
#     line_labels.append((line, label))
#
# ax.axhline(1, color='r', linestyle='--', linewidth=1, label='Stability limits')
# ax.axhline(-1, color='r', linestyle='--', linewidth=1)
# ax.fill_between(d_curved * 1e3, -1, 1, color='green', alpha=0.1)
# ax.set_xlabel('Curved mirror distance $d_c$ (mm)')
# ax.set_ylabel('Stability parameter $s$')
# ax.set_title('Stability parameter vs $d_c$ for different crystal lengths')
#
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
# line_legend = ax.legend(
#     handles=[line[0] for line in line_labels],
#     labels=[line[1] for line in line_labels],
#     loc='center left',
#     bbox_to_anchor=(1, 0.8)
# )
#
# L_line = Line2D([0], [0], color='white', lw=0)
# R_line = Line2D([0], [0], color='white', lw=0)
# wavelength_line = Line2D([0], [0], color='white', lw=0)
#
# param_legend = ax.legend(
#     handles=[L_line, R_line, wavelength_line],
#     labels=[f'L = {L} m', f'R = {R} m', f'Wavelength = {settings.wavelength * 1e9} nm'],
#     loc='center left',
#     bbox_to_anchor=(1, 0.15),
#     handlelength=0,
#     title='Used Parameters',
#     title_fontsize='medium'
# )
# param_legend.get_title().set_fontweight('bold')
# param_legend._legend_box.align = "left"
#
# stability_limit_line = Line2D([0], [0], color='r', linestyle='--', linewidth=1)
# stable_region_patch = plt.Line2D([0], [0], color='green', alpha=0.1, lw=10)
# max_waist_dot = Line2D([0], [0], color='black', marker='o', markersize=6, linestyle='None')
#
# indication_legend = ax.legend(
#     handles=[stability_limit_line, stable_region_patch, max_waist_dot],
#     labels=['Stability bounds: s = ±1', 'Stable region', 'Max waist'],
#     loc='upper left',
#     bbox_to_anchor=(1, 0.6)
# )
#
# ax.add_artist(line_legend)
# ax.add_artist(param_legend)
# ax.add_artist(indication_legend)
# plt.subplots_adjust(right=0.7)
# ax.grid(True)
# plt.show()
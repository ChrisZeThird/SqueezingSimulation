import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # For creating proxy artists

from cavity.cavity_formulas import ABCD_Matrix, stability_condition
from utils.settings import settings
import utils.plot_parameters

# Constants
L = settings.fixed_length
R = settings.R
d_curved = np.linspace(start=settings.d_curved_min, stop=settings.d_curved_max, num=settings.number_points)
l_crystal_array = np.array([10, 15, 20, 25, 30]) * 1e-3  # mm to meters

# Plotting stability parameter s vs d_curved for different crystal lengths
fig, ax = plt.subplots(figsize=(12, 6))  # Increased width for more space

# First legend data
line_labels = []

# Plot for each crystal length
for lc in l_crystal_array:
    d_valid, s_valid, w_valid = stability_condition(
        d_curved=d_curved,
        L=L,
        R=R,
        l_crystal=lc,
        index_crystal=settings.crystal_index,
        wavelength=settings.wavelength
    )

    if d_valid.size == 0:
        continue

    max_idx = np.argmax(w_valid)
    d_max = d_valid[max_idx] * 1e3  # mm
    s_max = s_valid[max_idx]
    w_max = w_valid[max_idx] * 1e6  # µm

    label = f'$l_{{c}}$ = {lc*1e3:.0f} mm, $w_{{max}}$ = {w_max:.1f} µm'
    line, = ax.plot(d_valid * 1e3, s_valid, label=label)
    color = line.get_color()

    # Mark the point (optional)
    ax.plot(d_max, s_max, 'o', color=color, markersize=6)

    # Add label for the line
    line_labels.append((line, label))

# Stability bounds
ax.axhline(1, color='r', linestyle='--', linewidth=1, label='Stability limits')
ax.axhline(-1, color='r', linestyle='--', linewidth=1)
ax.fill_between(d_curved * 1e3, -1, 1, color='green', alpha=0.1)

# Labels and layout
ax.set_xlabel('Curved mirror distance $d_c$ (mm)')
ax.set_ylabel('Stability parameter $s$')
ax.set_title('Stability parameter vs $d_c$ for different crystal lengths')

# Shrink current axis by 20% to make room for the legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# First legend block for the lines (crystal length and waist)
line_legend = ax.legend(
    handles=[line[0] for line in line_labels],  # Lines from the previous loop
    labels=[line[1] for line in line_labels],   # Labels from the previous loop
    loc='center left',
    bbox_to_anchor=(1, 0.8)
)

# Second legend block for used parameters (L, R, wavelength)
# Create proxy artists for parameters
L_line = Line2D([0], [0], color='white', lw=0)
R_line = Line2D([0], [0], color='white', lw=0)
wavelength_line = Line2D([0], [0], color='white', lw=0)

param_legend = ax.legend(
    handles=[L_line, R_line, wavelength_line],
    labels=[f'L = {L} m', f'R = {R} m', f'Wavelength = {settings.wavelength * 1e9} nm'],
    loc='center left',
    bbox_to_anchor=(1, 0.15),
    handlelength=0,  # Set handlelength to 0 to prevent invisible lines from showing up
    title='Used Parameters',  # Add a small title to the legend
    title_fontsize='medium',  # Set the font size for the title (optional)
)
param_legend.get_title().set_fontweight('bold')  # Make the title bold
param_legend._legend_box.align = "left"

# Third legend block for indication (stability bounds and max waist)
stability_limit_line = Line2D([0], [0], color='r', linestyle='--', linewidth=1)
stable_region_patch = plt.Line2D([0], [0], color='green', alpha=0.1, lw=10)
max_waist_dot = Line2D([0], [0], color='black', marker='o', markersize=6, linestyle='None')

# Third legend block for indication
indication_legend = ax.legend(
    handles=[stability_limit_line, stable_region_patch, max_waist_dot],
    labels=['Stability bounds: s = ±1', 'Stable region', 'Max waist'],
    loc='upper left',
    bbox_to_anchor=(1, 0.6)
)

# Add all legends to the plot
ax.add_artist(line_legend)
ax.add_artist(param_legend)
ax.add_artist(indication_legend)

# Adjust layout to avoid cutting off content (add padding on the right)
plt.subplots_adjust(right=0.7)

# Plot the grid and tighten layout
ax.grid(True)

# Show the plot
plt.show()

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

from nlo.sellmeier import n_z

from utils.tools_db import load_all_data
from utils.settings import settings
import utils.plot_parameters


def safe_divide(array, value):
    """
    Some values might be missing in articles like input powers. Adds safeguards for ratio with missing values
    :param array:
    :param value:
    :return:
    """
    try:
        if value is None or not np.isscalar(value):
            raise ValueError
        return array / float(value)
    except (KeyError, TypeError, ValueError):
        return np.full_like(array, np.nan, dtype=np.float64)


def safe_multiply(*args):
    # Replace non-numeric inputs with np.nan
    def safe_to_float(arg):
        try:
            # Attempt conversion to float
            return np.float64(arg)
        except (ValueError, TypeError):
            # If conversion fails, return np.nan
            return np.nan

    # Apply safe conversion to each argument
    args = [safe_to_float(arg) for arg in args]

    # If any argument is np.nan, the result will be np.nan
    result = args[0]
    for arg in args[1:]:
        result = np.where(np.isnan(result) | np.isnan(arg), np.nan, result * arg)

    return result


db_data = load_all_data()  # load the database

# Containers for filtered data
shg_entries = {}
opo_entries = {}
# Iterate over all data
for author, subsystems in db_data.items():
    for system_name, entries in subsystems.items():
        if not entries:
            continue
        key = f"{author}:{system_name.lower()}"
        if "shg" in system_name.lower():
            shg_entries[key] = entries
        elif "opo" in system_name.lower():
            opo_entries[key] = entries

# Get all authors who have OPO entries
authors = [key.split(":")[0] for key in opo_entries.keys()]
author_colors = {author: plt.cm.tab10(i % 10) for i, author in enumerate(authors)}

# Set our experiment parameters
wavelength = 0.780  # µm
input_power = np.linspace(start=20, stop=150, num=100)  # mW
length_crystal = np.array([10, 20, 25, 30])  # mm
roc = np.array([50, 75, 100, 150])
index_780 = n_z(wavelength)
index_390 = n_z(wavelength/2)
round_trip = np.linspace(start=450, stop=600, num=100)
output_coupler = np.array([0.01, 0.02, 0.05, 0.07, 0.10, 0.12, 0.15])

# Create figure and grid layout
fig = plt.figure()
gs = GridSpec(3, 2, height_ratios=[1, 1, 0.2], hspace=0.4)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1])
]

# Caption axis
caption_ax = fig.add_subplot(gs[2, :])
caption_ax.axis("off")

handles, labels = [], []

for author in authors:
    key = f"{author}:opo"
    if key not in opo_entries:
        continue
    opo = opo_entries[key][0]

    wavelength_ref_opo = opo["input_wavelength_nm"] * 1e-3
    index_ref2 = n_z(wavelength_ref_opo * 2)
    index_ratio = index_780 / index_ref2

    # Extract parameters for the legend
    wavelength_ref_opo = opo["input_wavelength_nm"] * 1e-3
    input_power_mW = opo.get("input_power_mW")
    crystal_length_mm = opo.get("crystal_length_mm")
    T_output_coupler = opo.get("T_output_coupler")
    cavity_length_mm = opo.get("cavity_length_mm")
    roc1_mm = opo.get("roc1_mm")

    power_ratio = safe_divide(input_power, input_power_mW)
    coupler_ratio = safe_divide(output_coupler, T_output_coupler)
    crystal_ratio = safe_divide(length_crystal, crystal_length_mm)
    round_trip_ratio = safe_divide(round_trip, cavity_length_mm)
    roc_ratio = safe_divide(roc, roc1_mm)

    S = 10 ** (opo["squeezing_dB"] / 10)  # sq (dB) = 10 log10 ( ) => S = 10^( sq (dB) / 10 )
    expected_squeezing_power = safe_multiply(S, index_ratio, power_ratio)
    expected_squeezing_crystal = safe_multiply(S, index_ratio, crystal_ratio)
    expected_squeezing_coupler = safe_multiply(S, index_ratio, coupler_ratio)
    expected_squeezing_roc = safe_multiply(S, index_ratio, roc_ratio)

    color = author_colors[author]
    line_power, = axes[0].plot(input_power, expected_squeezing_power, label=author, color=color)
    line_crystal, = axes[1].plot(length_crystal, expected_squeezing_crystal, '-s', label=author, color=color)
    line_coupler, = axes[2].plot(output_coupler, expected_squeezing_coupler, label=author, color=color)
    line_roc, = axes[3].plot(roc, expected_squeezing_roc, '-s', label=author, color=color)

    legend_label = (
        fr"$\mathbf{{{author}}}$: " + "\n"
        fr"$\lambda={wavelength_ref_opo:.3f}~\mu\text{{m}}$ " + "\n"
        fr"$P={input_power_mW}~\text{{mW}}$" + "\n"
        fr"$l_c={crystal_length_mm}~\text{{mm}}$" + "\n"
        fr"$T={T_output_coupler}$" + "\n"
        fr"$L={cavity_length_mm}~\text{{mm}}$"+ "\n"
        fr"$\text{{ROC}}={roc1_mm}~\text{{mm}}$" + "\n"
    )

    if author not in labels:
        handles.append(line_power)
        labels.append(legend_label)

# Titles
axes[0].set_title("1) Squeezing vs. Input Power (mW)", fontsize=20)
axes[1].set_title("2) Squeezing vs. Crystal Length (mm)", fontsize=20)
axes[2].set_title("3) Squeezing vs. Output Coupler", fontsize=20)
axes[3].set_title("4) Squeezing vs. Mirror ROC (mm)", fontsize=20)

# Legend (centered below plots)
fig.subplots_adjust(bottom=0.15)  # Adjust this value as needed
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=len(authors), fontsize=13)

# Caption
# caption = (
#     r"The expected squeezing at 780 nm is estimated using: "
#     r"$S_{780} = S_{\lambda} \times R_P \times R_L \times R_C \times R_R \times R_n$, "
#     "where each R represents the ratio between the value at 780 nm and the reference value from each article "
#     "for input power, crystal length, output coupler, and mirror ROC."
# )
# caption_ax.text(0.5, 0.5, caption, ha="center", va="center", fontsize=12, wrap=True)

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.show()

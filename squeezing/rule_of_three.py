import numpy as np

import matplotlib.pyplot as plt

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

# Setup plots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
colors = plt.cm.tab10(np.linspace(0, 1, len(authors)))  # Distinct colors

# Collect handles and labels for the legend
handles, labels = [], []

for author in authors:
    key = f"{author}:opo"
    if key not in opo_entries:
        continue
    opo = opo_entries[key][0]
    # print(opo)
    # Compute index ratios
    wavelength_ref_opo = opo["input_wavelength_nm"] * 1e-3  # µm
    index_ref2 = n_z(wavelength_ref_opo * 2)
    index_ratio = index_780 / index_ref2

    # Compute safe ratios
    power_ratio = safe_divide(input_power, opo.get("input_power_mW"))
    coupler_ratio = safe_divide(output_coupler, opo.get("T_output_coupler"))
    crystal_ratio = safe_divide(length_crystal, opo.get("crystal_length_mm"))
    round_trip_ratio = safe_divide(round_trip, opo.get("cavity_length_mm"))
    roc_ratio = safe_divide(roc, opo.get("roc1_mm"))

    # print(power_ratio)
    # Expected squeezing
    S = opo["squeezing_dB"]
    expected_squeezing_power = safe_multiply(S, index_ratio, power_ratio)
    expected_squeezing_crystal = safe_multiply(S, index_ratio, crystal_ratio)
    expected_squeezing_coupler = safe_multiply(S, index_ratio, coupler_ratio)
    expected_squeezing_roc = safe_multiply(S, index_ratio, roc_ratio)

    # Plot each line for subplots
    color = author_colors[author]
    line_power, = axes[0, 0].plot(input_power, expected_squeezing_power, label=author, color=color)
    line_crystal, = axes[0, 1].plot(length_crystal, expected_squeezing_crystal, '-s', label=author, color=color)
    line_coupler, = axes[1, 0].plot(output_coupler, expected_squeezing_coupler, label=author, color=color)
    line_roc, = axes[1, 1].plot(roc, expected_squeezing_roc, '-s', label=author, color=color)

    # Append just one line per author to legend handles
    if author not in labels:
        handles.append(line_power)
        labels.append(author)

    # # Mark reference points
    # axes[0, 0].plot(opo["input_power_mW"], 3, 'ko')
    # axes[0, 1].plot(opo["crystal_length_mm"], 3, 'ko')
    # axes[1, 0].plot(opo["T_output_coupler"], 3, 'ko')
    # axes[1, 1].plot(opo["roc1_mm"], 3, 'ko')

# Add titles and legends
axes[0, 0].set_title("1) Squeezing vs. Input Power (mW)")
axes[0, 1].set_title("2) Squeezing vs. Crystal Length (mm)")
axes[1, 0].set_title("3) Squeezing vs. Output Coupler")
axes[1, 1].set_title("4) Squeezing vs. Mirror ROC (mm)")

# Add a single legend for the entire figure
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=12)

# Add caption
caption = (
    r"The expected squeezing at 780 nm is estimated using: "
    r"$S_{780} = S_{\lambda} \times R_P \times R_L \times R_C \times R_R \times R_n$, "
    "where each R represents the ratio between the value at 780 nm and the reference value from each article "
    "for input power, crystal length, output coupler, and mirror ROC. "
    "Black circles show reference points from each paper."
)

plt.figtext(0.02, 0.01, caption, wrap=True, fontsize=13, ha='left')
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()

# opo = opo_entries["burks:opo"][0]
#
# wavelength_ref_opo = opo["input_wavelength_nm"] * 1e-3  # µm
# index_wavelength_ref = n_z(wavelength_ref_opo)
# index_wavelength_ref2 = n_z(wavelength_ref_opo * 2)
#
# # define ratios
# crystal_ratio = length_crystal / opo["crystal_length_mm"]
# round_trip_ratio = round_trip / opo["cavity_length_mm"]
# roc_ratio = roc / opo["roc1_mm"]  # assume roc is the same for both mirrors
# index_ratio = index_780 / index_wavelength_ref2
# power_ratio = input_power / opo["input_power_mW"]
# coupler_ratio = output_coupler / opo["T_output_coupler"]
#
# # Calculate expected squeezing
# expected_squeezing_power = opo["squeezing_dB"] * index_ratio * power_ratio
# expected_squeezing_crystal = opo["squeezing_dB"] * index_ratio * crystal_ratio
# # expected_squeezing_roundtrip = opo["squeezing_dB"] * index_ratio * round_trip_ratio
# expected_squeezing_coupler = opo["squeezing_dB"] * index_ratio * coupler_ratio
# expected_squeezing_roc = opo["squeezing_dB"] * index_ratio * roc_ratio
#
# # Create 2x2 subplots
# fig, axes = plt.subplots(2, 2, figsize=(16, 10))
#
# # Masks to avoid double plotting
# mask_power = input_power != opo["input_power_mW"]
# mask_crystal = length_crystal != opo["crystal_length_mm"]
# mask_coupler = output_coupler != opo["T_output_coupler"]
# mask_roc = roc != opo["roc1_mm"]
#
# # Subplot 1: input power
# axes[0, 0].plot(input_power[mask_power], expected_squeezing_power[mask_power])
# axes[0, 0].plot(opo["input_power_mW"], 3, 'ks')
# axes[0, 0].set_title("1) Squeezing vs. Input Power (mW)")
#
# # Subplot 2: crystal length
# axes[0, 1].plot(length_crystal[mask_crystal], expected_squeezing_crystal[mask_crystal], '-s')
# axes[0, 1].plot(opo["crystal_length_mm"], 3, 'ks')
# axes[0, 1].set_title("2) Squeezing vs. Crystal Length (mm)")
#
# # Subplot 3: output coupler
# axes[1, 0].plot(output_coupler[mask_coupler], expected_squeezing_coupler[mask_coupler])
# axes[1, 0].plot(opo["T_output_coupler"], 3, 'ks')
# axes[1, 0].set_title("3) Squeezing vs. Output Coupler")
#
# # Subplot 4: mirror ROC
# axes[1, 1].plot(roc[mask_roc], expected_squeezing_roc[mask_roc], '-s')
# axes[1, 1].plot(opo["roc1_mm"], 3, 'ks')
# axes[1, 1].set_title("4) Squeezing vs. Mirror ROC (mm)")
#
# # Caption with formula and graph explanations
# caption = (
#     r"The expected squeezing at 780 nm is estimated using: "
#     r"$S_{780} = S_{\lambda} \times R_P \times R_L \times R_C \times R_R \times R_n$, "
#     "where each R represents the ratio between the value at 780 nm and the value used in the reference article "
#     "(for input power, crystal length, cavity geometry, mirror curvature, etc.), "
#     "and $R_n$ is the fixed ratio of refractive indices. "
#     "In graph 1), we vary $P_{blue}$; in 2), the crystal length; "
#     "in 3), the output coupler transmission; and in 4), the mirror radius of curvature. "
#     "All other parameters are fixed to the reference point (black marker)."
# )
#
# plt.figtext(0.02, 0.01, caption, wrap=True, fontsize=13, ha='left')
# plt.tight_layout(rect=[0, 0.07, 1, 1])  # leave space for caption
# plt.show()

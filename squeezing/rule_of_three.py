import numpy as np

import matplotlib.pyplot as plt

from nlo.sellmeier import n_z

from utils.tools_db import load_all_data
from utils.settings import settings
import utils.plot_parameters

db_data = load_all_data()  # load the database

# Containers for filtered data
shg_entries = {}
opo_entries = {}
# Iterate over all data
for author, subsystems in db_data.items():
    for system_name, entries in subsystems.items():
        if not entries:
            continue  # skip empty systems

        key = f"{author}:{system_name.lower()}"

        # Identify by system name
        if "shg" in system_name.lower():
            shg_entries[key] = entries
        elif "opo" in system_name.lower():
            opo_entries[key] = entries

# Example print
for key, entries in shg_entries.items():
    print(f"SHG entries for {key}:")
    for entry in entries:
        print(f"  ID {entry['id']}: {entry}")

for key, entries in opo_entries.items():
    print(f"OPO entries for {key}:")
    for entry in entries:
        print(f"  ID {entry['id']}: {entry}")

# Set our experiment parameters
wavelength = 0.780  # µm
input_power = np.linspace(start=20, stop=150, num=100)  # mW
length_crystal = np.array([10, 20, 25, 30])  # mm
roc = np.array([50, 75, 100, 150])
index_780 = n_z(wavelength)
index_390 = n_z(wavelength/2)
round_trip = np.linspace(start=450, stop=600, num=100)
output_coupler = np.array([0.01, 0.02, 0.05, 0.07, 0.10, 0.12, 0.15])

opo = opo_entries["burks:opo"][0]

wavelength_ref_opo = opo["input_wavelength_nm"] * 1e-3  # µm
index_wavelength_ref = n_z(wavelength_ref_opo)
index_wavelength_ref2 = n_z(wavelength_ref_opo * 2)

# define ratios
crystal_ratio = length_crystal / opo["crystal_length_mm"]
round_trip_ratio = round_trip / opo["cavity_length_mm"]
roc_ratio = roc / opo["roc1_mm"]  # assume roc is the same for both mirrors
index_ratio = index_780 / index_wavelength_ref2
power_ratio = input_power / opo["input_power_mW"]
coupler_ratio = output_coupler / opo["T_output_coupler"]

# Calculate expected squeezing
expected_squeezing_power = opo["squeezing_dB"] * index_ratio * power_ratio
expected_squeezing_crystal = opo["squeezing_dB"] * index_ratio * crystal_ratio
# expected_squeezing_roundtrip = opo["squeezing_dB"] * index_ratio * round_trip_ratio
expected_squeezing_coupler = opo["squeezing_dB"] * index_ratio * coupler_ratio
expected_squeezing_roc = opo["squeezing_dB"] * index_ratio * roc_ratio

# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Masks to avoid double plotting
mask_power = input_power != opo["input_power_mW"]
mask_crystal = length_crystal != opo["crystal_length_mm"]
mask_coupler = output_coupler != opo["T_output_coupler"]
mask_roc = roc != opo["roc1_mm"]

# Subplot 1: input power
axes[0, 0].plot(input_power[mask_power], expected_squeezing_power[mask_power])
axes[0, 0].plot(opo["input_power_mW"], 3, 'ks')
axes[0, 0].set_title("1) Squeezing vs. Input Power (mW)")

# Subplot 2: crystal length
axes[0, 1].plot(length_crystal[mask_crystal], expected_squeezing_crystal[mask_crystal], '-s')
axes[0, 1].plot(opo["crystal_length_mm"], 3, 'ks')
axes[0, 1].set_title("2) Squeezing vs. Crystal Length (mm)")

# Subplot 3: output coupler
axes[1, 0].plot(output_coupler[mask_coupler], expected_squeezing_coupler[mask_coupler])
axes[1, 0].plot(opo["T_output_coupler"], 3, 'ks')
axes[1, 0].set_title("3) Squeezing vs. Output Coupler")

# Subplot 4: mirror ROC
axes[1, 1].plot(roc[mask_roc], expected_squeezing_roc[mask_roc], '-s')
axes[1, 1].plot(opo["roc1_mm"], 3, 'ks')
axes[1, 1].set_title("4) Squeezing vs. Mirror ROC (mm)")

# Caption with formula and graph explanations
caption = (
    r"The expected squeezing at 780 nm is estimated using: "
    r"$S_{780} = S_{\lambda} \times R_P \times R_L \times R_C \times R_R \times R_n$, "
    "where each R represents the ratio between the value at 780 nm and the value used in the reference article "
    "(for input power, crystal length, cavity geometry, mirror curvature, etc.), "
    "and $R_n$ is the fixed ratio of refractive indices. "
    "In graph 1), we vary $P_{blue}$; in 2), the crystal length; "
    "in 3), the output coupler transmission; and in 4), the mirror radius of curvature. "
    "All other parameters are fixed to the reference point (black marker)."
)

plt.figtext(0.02, 0.01, caption, wrap=True, fontsize=13, ha='left')
plt.tight_layout(rect=[0, 0.07, 1, 1])  # leave space for caption
plt.show()

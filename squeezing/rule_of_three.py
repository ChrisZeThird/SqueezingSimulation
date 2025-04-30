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
input_power = np.linspace(start=20, stop=200, num=100)  # mW
length_crystal = np.array([10, 20, 25, 30])  # mm
index_780 = n_z(wavelength)
index_390 = n_z(wavelength/2)

opo = opo_entries["burks:opo"][0]

wavelength_ref_opo = opo["input_wavelength_nm"] * 1e-3  # µm
index_wavelength_ref = n_z(wavelength_ref_opo)
index_wavelength_ref2 = n_z(wavelength_ref_opo * 2)

# define ratios
crystal_ratio = length_crystal[0] / opo["crystal_length_mm"]
index_ratio = index_780 / index_wavelength_ref2
power_ratio = input_power / opo["input_power_mW"]

expected_squeezing = opo["squeezing_dB"] * power_ratio * index_ratio

fig = plt.figure(figsize=(16, 9))
plt.plot(input_power, expected_squeezing)
plt.xlabel("input power (mW)")
plt.ylabel("expected squeezing (dB)")
plt.show()







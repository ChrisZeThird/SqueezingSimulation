import numpy as np
import matplotlib.pyplot as plt

from cavity.cavity_formulas import Beam_waist
from utils.settings import settings

# Thermal coefficients
alpha = 0.186  # Thermal expansion coefficient in 1/K
Kc = 13  # Thermal conductivity in W/(m*K)
dn_dT = 1.6e-5  # dn/dT in 1/K
P = 1  # Power in W

# ABCD matrix parameters
d_curved = np.linspace(start=settings.d_curved_min, stop=settings.d_curved_max, num=settings.number_points)
z1, z2, w1, w2, (valid_z1, valid_z2) = Beam_waist(d_curved=d_curved,
                                                  L=settings.fixed_length,
                                                  R=settings.R,
                                                  l_crystal=settings.crystal_length)


def thermal_lensing(Kc=Kc, P=P, alpha=alpha, dn=dn_dT, waist=w1):
    """
    Calculate the thermal lensing effect based on the parameters provided.
    :param Kc:
    :param P:
    :param alpha:
    :param dn:
    :param waist:
    :return:
    """
    constant = Kc * np.pi / (P * dn * (1 - np.exp(- alpha * settings.crystal_length)))
    f_th = constant * waist ** 2
    return f_th


# Plot the thermal lensing effect against the beam waist
f_th = thermal_lensing(waist=w1[valid_z1])

plt.figure(figsize=(10, 6))
plt.plot(w1[valid_z1], f_th, label='Thermal Lens Effect', color='blue')
plt.xlabel('Beam Waist (m)')
plt.ylabel('Thermal Lens Focal Length (m)')
plt.title('Thermal Lensing Effect vs Beam Waist')
plt.grid()
plt.legend()
plt.show()
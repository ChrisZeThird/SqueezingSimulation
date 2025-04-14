from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt

import numpy as np

from utils.settings import settings
import utils.plot_parameters

import gaussian.propagation as prop
import gaussian.ray_matrix as rm

# Define minimum waist
w0 = 0.2  # mm

# Set all distances to mm
wavelength = settings.wavelength  # mm
L = settings.fixed_length
d_curved = settings.fixed_d_curved
l_crystal = settings.crystal_length
R = settings.R
crystal_index = settings.crystal_index

dc_optical = d_curved + l_crystal * (crystal_index - 1)
print(dc_optical)
# dc_optical = 115

M = rm.M_free_space(dc_optical/2) @ rm.M_focal(R/2) @ rm.M_free_space(L - dc_optical) @ rm.M_focal(R/2) @ rm.M_free_space(dc_optical/2)
A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

print('A:', A, 'B', B)
print('C:', C, 'D', D)

q_imag = np.sqrt(1 - A**2) / C
# q_image = np.sqrt(-C * B)
print(q_imag)

w = np.sqrt((wavelength/np.pi) * q_imag)

print('waist: ', w)

# # Calculates parameters
# zR = prop.rayleigh_range(w=w0, wavelength=wavelength)
# print('Rayleigh Range: ', zR)
# print('----------------------')
#
# q = prop.q_parameter(z=0, w0=w0, wavelength=wavelength)
#
# m1 = rm.M1(L=L, d_curved=d_curved, l_crystal=l_crystal, index_crystal=settings.crystal_index, R=R)
# m2 = rm.M2(L=L, d_curved=d_curved, l_crystal=l_crystal, index_crystal=settings.crystal_index, R=R)
# print('{m1, m2}', m1, m2)
# print('----------------------')
#
# q1 = prop.propagate(q, m1)
# q2 = prop.propagate(q, m2)
# print('{q1, q2}: ', (q1, q2))
# print('----------------------')

# w1 = prop.waist(q1, wavelength=wavelength)
# w2 = prop.waist(q2, wavelength=wavelength)
# w1, w2 = prop.waist(wavelength=wavelength, n=settings.crystal_index, prop_matrix=m1)
# print('w1, w2: ', (w1, w2))
# print('----------------------')

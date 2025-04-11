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
wavelength = settings.wavelength * 1e3  # mm
L = settings.fixed_length * 1e3
d_curved = settings.fixed_d_curved * 1e3
l_crystal = settings.crystal_length * 1e3
R = settings.R * 1e3

# Calculates parameters
zR = prop.rayleigh_range(w=w0, wavelength=wavelength)
print('Rayleigh Range: ', zR)
print('----------------------')

q = prop.q_parameter(z=0, w0=w0, wavelength=wavelength)

m1 = rm.M1(L=L, d_curved=d_curved, l_crystal=l_crystal, index_crystal=settings.crystal_index, R=R)
m2 = rm.M2(L=L, d_curved=d_curved, l_crystal=l_crystal, index_crystal=settings.crystal_index, R=R)
print('{m1, m2}', m1, m2)
print('----------------------')

q1 = prop.propagate(q, m1)
q2 = prop.propagate(q, m2)
print('{q1, q2}: ', (q1, q2))
print('----------------------')

w1 = prop.waist(q1, wavelength=wavelength)
w2 = prop.waist(q2, wavelength=wavelength)
print('w1, w2: ', (w1, w2))
print('----------------------')

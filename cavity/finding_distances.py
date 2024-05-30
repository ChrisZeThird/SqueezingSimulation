import matplotlib.pyplot as plt
import numpy as np


# Finding cavity distances
def cos_theta_function(R, d_curved, l):
    """

    :param R:
    :param d_curved:
    :param l:
    :return:
    """
    return 2 * (R/(d_curved - l))**2 - 1


def finding_unknown_distance(L, R, l, d_curved):
    """
    Calculate missing distances in a ring cavity configuration. Please ensure all lengths are in meters!
    :param L: round-trip length
    :param R: curved mirror radius
    :param l: crystal length
    :param d_curved: distance between curved mirrors, by default None
    :return: the unknown distances as well as the incident angles on the mirrors
    """
    cos_theta = cos_theta_function(R, d_curved, l)
    d_flat = L/(1 + (1/cos_theta)) - d_curved
    OF = d_flat / (2 * cos_theta)
    OC = d_curved / (2 * cos_theta)
    S = (d_curved - l)/2 + OC + OF

    return d_flat, OF, OC, cos_theta, S


# L = 500e-3
# d_curved = 57.7e-3
# R = 100e-3
# l = 10e-3
#
# d_flat, OF, OC, cos_theta, S = finding_unknown_distance(L, R, l, d_curved)
# print('Distance between the flat mirrors: ', d_flat)


def equivalent_cavity(d_flat, d_curved, l, R):
    """

    :param d_flat:
    :param d_curved:
    :param l:
    :param R:
    :return:
    """
    cos_theta = cos_theta_function(R, d_curved, l)
    return d_flat / (2 * cos_theta) + d_curved / (2 * cos_theta) + (d_curved - l) / 2


def crystal_waist(L, R, d_curved, l, wavelength=780e-9):
    """
    Compute the beam waist on the crystal according to O. Pinel. 'Optique quantique multimode avec des peignes de
    fréquence' PhD thesis, 2010.
    :param L: total cavity length
    :param R: curved mirror radius
    :param d_curved: distance between curved mirrors
    :param l: crystal length
    :param wavelength:
    :return:
    """
    d_flat, _, _, _, S = finding_unknown_distance(L, R, l, d_curved)
    # S = equivalent_cavity(d_flat, d_curved, l, R)
    L_prime = (d_curved - l)/2
    A = (L_prime - R / 2)
    B = ((S - L_prime) * L_prime - S * R / 2)
    C = (S - L_prime - R / 2)
    temp = A * B / C
    index = np.where((temp < 0))
    # print(A, B, C)
    return np.sqrt(wavelength / np.pi) * ((- A[index] * B[index]) / C[index]) ** (1 / 4), index


# print(crystal_waist(L=L, R=R, d_curved=d_curved, l=l, wavelength=860e-9))

""" Checking some parameters """
# radii = np.array([25, 50, 100]) * 1e-3
# crystal_lengths = np.arange(start=5, stop=30, step=5) * 1e-3
# expected_waists = {"G. Hétet": 40e-6, "Appel": None, "T. Takahito": 20e-6, "Takao": 17e-6}

# distance_crystal_mirror = (d_curved - l)/2
# print('Crystal waist: ', crystal_waist(L=distance_crystal_mirror, R=R, S=S, wavelength=795e-9))

# R = 50e-3
# L = 500e-3
# d_curved = np.linspace(0, 100e-3, num=200)
# crystal_length = 15e-3
# wavelength = 860e-6
#
# waist, index = crystal_waist(L, R, d_curved, crystal_length, wavelength)

# plt.plot(d_curved[index], waist)
# plt.show()

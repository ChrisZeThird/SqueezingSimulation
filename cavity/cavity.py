import numpy as np
import matplotlib.pyplot as plt

""" Finding cavity distances """


def finding_unknown_distance(L, R, l, d_curved):
    """
    Calculate missing distances in a ring cavity configuration. Please ensure all lengths are in meters!
    :param L: round-trip length
    :param R: curved mirror radius
    :param l: crystal length
    :param d_curved: distance between curved mirrors, by default None
    :return: the unknown distances as well as the incident angles on the mirrors
    """
    if d_curved is None:
        raise ValueError(
            'Both distances between flat mirrors and curved mirrors cannot be unknown. Please fix the distance between the curved mirror')

    else:
        cos_theta = 2 * (R/(d_curved - l))**2 - 1
        d_flat = L/(1 + (1/cos_theta)) - d_curved
        OF = d_flat / (2 * cos_theta)
        OC = d_curved / (2 * cos_theta)
        S = (d_curved - l)/2 + OC + OF
        print('Calculated total length L: ', d_flat + d_curved + 2 * (OC + OF))
        print('Input total length L: ', L)
        print('--------------------------')
        print('Calculated equivalent cavity length S: ', S)
        return d_flat, OF, OC, cos_theta, S


L = 64e-2
d_curved = 11.6e-2
R = 10e-2
l = 10e-3

d_flat, OF, OC, cos_theta, S = finding_unknown_distance(L, R, l, d_curved)
print(d_flat)


def crystal_waist(L, R, S, wavelength=780e-9):
    """
    Compute the beam waist on the crystal according to O. Pinel. 'Optique quantique multimode avec des peignes de
    fréquence' PhD thesis, 2010.
    :param L: stability length
    :param R: curved mirror radius
    :param S: length of the equivalent cavity
    :param wavelength:
    :return:
    """
    return np.sqrt(wavelength / np.pi) * ((- (L - R / 2)((S - L) * L - S * R / 2)) / (S - L - R / 2)) ** (1 / 4)


""" Checking some parameters """
radii = np.array([25, 50, 100]) * 1e-3
crystal_lengths = np.arange(start=5, stop=30, step=5) * 1e-3
expected_waists = {"G. Hétet": 40e-6, "Appel": None, "T. Takahito": 20e-6, "Takao": 17e-6}

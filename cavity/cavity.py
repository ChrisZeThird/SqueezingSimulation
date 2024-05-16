import numpy as np
import matplotlib.pyplot as plt


def crystal_waist(L, R, S, wavelength):
    """
    Compute the beam waist on the crystal according to O. Pinel. 'Optique quantique multimode avec des peignes de
    fréquence' PhD thesis, 2010.
    :param L: stability length
    :param R: curved mirror radius
    :param S: length of the equivalent cavity
    :param wavelength:
    :return:
    """
    return np.sqrt(wavelength/np.pi) * ((- (L - R/2)((S - L) * L - S * R/2))/(S - L - R/2))**(1/4)


""" Checking some parameters """
radii = np.array([25, 50, 100]) * 1e-3
crystal_lengths = np.arange(start=5, stop=30, step=5) * 1e-3
expected_waists = {"G. Hétet": 40e-6, "Appel": None, "T. Takahito": 20e-6, "Takao": 17e-6}

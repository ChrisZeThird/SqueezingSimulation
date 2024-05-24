import numpy as np

import cavity.finding_distances as fd
from utils.settings import settings


# -- General cavity formulas -- #
def Finesse(T, Loss):
    """
    Cavity Finesse
    :param T: Transmission coefficient
    :param Loss: Intra-cavity Loss
    :return:
    """
    return np.pi * (((1 - T) * (1 - Loss)) ** (1/4)) / (1 - np.sqrt((1 - T) * (1 - Loss)))


def FSR(L):
    """
    Free Spectral Range (frequency domain)
    :param L: Cavity length
    :return:
    """
    return settings.c / L


def Bandwidth(T, Loss, L):
    """
    Calculate bandwidth of bow-tie ring cavity (frequency domain)
    :param T: Transmission coefficient
    :param Loss: Intra-cavity loss
    :param L: Cavity length
    :return:
    """
    return FSR(L=L) / Finesse(T=T, Loss=Loss)


def Escape_efficiency(T, Loss):
    """
    
    :param T: Transmission coefficient
    :param Loss: Intra-cavity loss
    :return: 
    """
    return T / (T + Loss)


def Pump_power(T, Loss, E):
    """

    :param T: Transmission coefficient
    :param Loss: Loss: Intra-cavity loss
    :param E: Effective non-linearity of optical medium
    :return:
    """
    return ((T + Loss) ** 2) / (4 * E)


# -- Ray propagation -- #
def ABCD_Matrix(d_curved, d_flat, d_diag, R, l_crystal, index_crystal=1):
    """

    :param d_curved: Distance between curved mirrors
    :param d_flat: Distance between flat mirrors
    :param d_diag: Distance between curved and flat mirrors (diagonal line)
    :param R: Radii of curvature of curved mirror
    :param l_crystal: Length of non-linear crystal
    :param index_crystal: Index of refraction of non-linear medium (by default 1)
    :return: Tuple (A, B, C, D)
    """
    E = (- l_crystal / 2) + ((d_curved - l_crystal) / 2) * (1 / index_crystal)
    F = ((- 2 / R) * E) + (1 / index_crystal)

    A = 1 + (d_diag + d_flat / 2) * (- 2 / R)
    B = E + (d_diag + d_flat / 2) * F
    C = - 2 / R
    D = F

    return A, B, C, D


def Rayleigh_length(A, B, C, D):
    """
    Calculate Rayleigh length from ABCD matrix elements of beam waist at the crystal center
    :param A:
    :param B:
    :param C:
    :param D:
    :return:
    """
    print(- (A * B) / (C * D) > 0)
    return np.sqrt(- (A * B) / (C * D))


def Beam_waist(d_curved, L, cavity_width, R, l_crystal, index_crystal=1, wavelength=780e-9):
    """
    Calculates the beam waist size in radius at the center of the nonlinear optical crystal and the intermediate
    between flat mirrors
    :param d_curved: Distance between curved mirrors
    :param L: Cavity length
    :param cavity_width:
    :param R: Radii of curvature of curved mirror
    :param l_crystal: Length of non-linear crystal
    :param index_crystal: Index of refraction of non-linear medium (by default 1)
    :param wavelength:
    :return: Tuple (w1, w2)
    """
    d_diag = finding_diagonal(L=L, cavity_width=cavity_width)
    d_flat = finding_flat(L=L, cavity_width=cavity_width, d_curved=d_curved)

    A, B, C, D = ABCD_Matrix(d_curved, d_flat, d_diag, R, l_crystal, index_crystal=index_crystal)
    rayleigh_length = Rayleigh_length(A, B, C, D)

    w1 = np.sqrt((wavelength / np.pi) * rayleigh_length)
    w2 = np.sqrt(A**2 + (B / rayleigh_length)**2) * w1

    return w1, w2


# -- Missing lengths -- #
def finding_diagonal(L, cavity_width):
    """

    :param L: Cavity length
    :param cavity_width:
    :return:
    """
    return (L / 4) + (cavity_width ** 2) / L


def finding_flat(L, cavity_width, d_curved):
    """

    :param L: Cavity length
    :param cavity_width:
    :param d_curved: Distance between curved mirrors
    :return:
    """
    return (L / 2) - 2 * (cavity_width ** 2) + d_curved

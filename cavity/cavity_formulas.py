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
    E = (l_crystal / (2 * index_crystal)) + ((d_curved - l_crystal) / 2) * (1 / index_crystal)
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
    :return: tuple with the Rayleigh length and the corresponding indices with allowed values
    """
    # print(- (A * B) / (C * D) > 0)
    product = - (A * B) / (C * D)
    temp = np.full(shape=A.shape, fill_value=np.nan, dtype=np.float32)

    valid_indices = np.where(product >= 0)
    temp[valid_indices] = np.sqrt(product[valid_indices])
    return temp[valid_indices], valid_indices


def Beam_waist(d_curved, L, cavity_width, R, l_crystal, index_crystal=1, wavelength=780e-9, tamagawa=False):
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
    :param tamagawa: Boolean, to calculate missing distances with the formula from Tamagawa Vol.2-3
    :return: Tuple (w1, w2, valid_indices)
    """
    if tamagawa:
        d_diag = finding_diagonal_tamagawa(L=L, cavity_width=cavity_width)
        d_flat = finding_flat_tamagawa(L=L, cavity_width=cavity_width, d_curved=d_curved)
    else:
        d_flat, OF, OC, _, _ = fd.finding_unknown_distance(L=L, R=R, l=l_crystal, d_curved=d_curved)
        d_diag = OF + OC

    A, B, C, D = ABCD_Matrix(d_curved, d_flat, d_diag, R, l_crystal, index_crystal=index_crystal)
    rayleigh_length, valid_indices = Rayleigh_length(A, B, C, D)

    w1 = np.sqrt((wavelength / np.pi) * rayleigh_length)
    # w2 = np.sqrt(A[valid_indices]**2 + (B[valid_indices] / rayleigh_length)**2) * w1
    w2 = (1 / (np.sqrt((C * rayleigh_length) ** 2 + D[valid_indices]))) * w1 / index_crystal

    return w1, w2, valid_indices


# -- Missing lengths -- #
def finding_diagonal_tamagawa(L, cavity_width):
    """

    :param L: Cavity length
    :param cavity_width:
    :return:
    """
    return (L / 4) + ((cavity_width ** 2) / L)


def finding_flat_tamagawa(L, cavity_width, d_curved):
    """

    :param L: Cavity length
    :param cavity_width:
    :param d_curved: Distance between curved mirrors
    :return:
    """
    return (L / 2) - (2 * (cavity_width ** 2)) + d_curved

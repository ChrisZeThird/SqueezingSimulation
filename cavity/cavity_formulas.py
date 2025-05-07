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


def Bandwidth_bowtie(T, Loss, L):
    """
    Calculate bandwidth of bow-tie ring cavity (frequency domain)
    :param T: Transmission coefficient
    :param Loss: Intra-cavity loss
    :param L: Cavity length
    :return:
    """
    return FSR(L=L) / Finesse(T=T, Loss=Loss)


def Bandwidth_linear(cavity_length, transmission_coefficient):
    """
    Calculates the bandwidth of a linear cavity from a couple (L, T)
    :param cavity_length: in meter
    :param transmission_coefficient:
    :return:
    """
    # return (c / (2 * cavity_length)) * (transmission_coefficient / (2 * np.pi))
    return (3e8 * transmission_coefficient) / (4 * np.pi * cavity_length)


def Escape_efficiency(T, Loss):
    """
    
    :param T: Transmission coefficient
    :param Loss: Intra-cavity loss
    :return: 
    """
    return T / (T + Loss)


def Pump_threshold(T, Loss, E):
    """
    Pump threshold power
    :param T: Transmission coefficient
    :param Loss: Loss: Intra-cavity loss
    :param E: Effective non-linearity of optical medium
    :return:
    """
    return ((T + Loss) ** 2) / (4 * E)


# -- Ray propagation -- #
def ABCD_Matrix(L, d_curved, R, l_crystal, index_crystal=settings.crystal_index):
    """
    Ray transfer matrix for a half single pass in a ring bow-tie cavity.

    :param L: Cavity round-trip length
    :param d_curved: Distance between curved mirrors
    :param R: Radii of curvature of curved mirror
    :param l_crystal: Length of non-linear crystal
    :param index_crystal: Index of refraction of non-linear medium (by default 1)
    :return: Tuple (A1, B1, C1, D1)
    """
    A1 = 1 - (L - d_curved) / R
    B1 = index_crystal * ((L - d_curved) / 2 + ((d_curved - l_crystal) / 2) * (1 - (L - d_curved) / R)) + l_crystal * (1 - (L - d_curved) / R) / 2
    C1 = - 2 / R
    D1 = (1 - (d_curved - l_crystal) / R) * index_crystal - l_crystal / R

    return A1, B1, C1, D1


# -- Tamagawa / Svelto -- #
def z_parameter(A1, B1, C1, D1):
    """
    Calculates z parameter
    :param A1:
    :param B1:
    :param C1:
    :param D1:
    :return:
    """
    z1 = - (B1 * D1) / (A1 * C1)  # wavefront at mirror 1
    z2 = - (A1 * B1) / (C1 * D1)  # wavefront at mirror 2

    return z1, z2


def compute_waist(z, wavelength, index):
    """
    Compute beam waist from the z-parameter, wavelength and index of refraction.
    Returns waist and valid indices where z >= 0.
    """
    temp = np.full_like(z, np.nan, dtype=np.float64)
    valid = z >= 0
    temp[valid] = np.sqrt(z[valid])
    waist = np.sqrt((wavelength / (index * np.pi)) * temp)
    return waist, valid


def stability_condition(d_curved, L, R, l_crystal, index_crystal=settings.crystal_index, wavelength=settings.wavelength):
    """
    Compute the stability condition.
    :return: (valid_d_curved, stability_value_s, waist_in_crystal)
    """
    A1, B1, C1, D1 = ABCD_Matrix(L=L, d_curved=d_curved, R=R, l_crystal=l_crystal, index_crystal=index_crystal)
    z1, _ = z_parameter(A1, B1, C1, D1)
    s = 2 * A1 * D1 - 1

    w1, valid_z = compute_waist(z1, wavelength, index_crystal)
    valid = np.logical_and(valid_z, np.abs(s) < 1)

    return d_curved[valid], s[valid], w1[valid]


def Beam_waist(d_curved, L, R, l_crystal, index_crystal=settings.crystal_index, wavelength=settings.wavelength):
    """
    Calculates the beam waist sizes in the crystal and in air.
    :return: Tuple (z1, z2, w1, w2, valid_indices)
    """
    A1, B1, C1, D1 = ABCD_Matrix(L=L, d_curved=d_curved, R=R, l_crystal=l_crystal, index_crystal=index_crystal)
    z1, z2 = z_parameter(A1, B1, C1, D1)

    w1, valid_z1 = compute_waist(z1, wavelength, index_crystal)
    w2, valid_z2 = compute_waist(z2, wavelength, 1.0)

    return z1, z2, w1, w2, (valid_z1, valid_z2)

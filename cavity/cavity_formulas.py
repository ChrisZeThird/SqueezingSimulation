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
def ABCD_Matrix(L, d_curved, R, l_crystal, index_crystal=1):
    """
    Ray transfer matrix for a single pass in a ring bow-tie cavity.

    :param L: Cavity round-trip length
    :param d_curved: Distance between curved mirrors
    :param R: Radii of curvature of curved mirror
    :param l_crystal: Length of non-linear crystal
    :param index_crystal: Index of refraction of non-linear medium (by default 1)
    :return: Tuple (A, B, C, D)
    """

    E = (L - d_curved) / 2

    A1 = 1 - E / R
    B1 = E + (d_curved - l_crystal) / 2 * (1 - E * (2 / R)) + l_crystal * (1 - E * (2 / R))/(2 * index_crystal)
    C1 = - 2 / R
    D1 = 1 - (d_curved - l_crystal) / R - l_crystal / (index_crystal * R)
    # D1 = C1 * (l_crystal / (2 * index_crystal) - (d_curved - l_crystal) / 2) + 1

    return A1, B1, C1, D1


# -- Tamagawa / Svelto -- #
def q_parameter(A, B, C, D):
    """
    Calculates q parameter
    :param A:
    :param B:
    :param C:
    :param D:
    :return:
    """
    q1 = - (B * D) / (A * C)  # wavefront at mirror 1
    q2 = - (B * A) / (D * C)  # wavefront at mirror 2

    return q1, q2


def Beam_waist(d_curved, L, R, l_crystal, index_crystal=1, wavelength=780e-9):
    """
    Calculates the beam waist size in radius at the center of the nonlinear optical crystal and the intermediate
    between flat mirrors
    :param d_curved: Distance between curved mirrors
    :param L: Cavity length
    :param R: Radii of curvature of curved mirror
    :param l_crystal: Length of non-linear crystal
    :param index_crystal: Index of refraction of non-linear medium (by default 1)
    :param wavelength:
    :return: Tuple (w1, w2, valid_indices)
    """
    A1, B1, C1, D1 = ABCD_Matrix(L=L, d_curved=d_curved, R=R, l_crystal=l_crystal, index_crystal=index_crystal)

    q1, q2 = q_parameter(A1, B1, C1, D1)

    temp1 = np.full(shape=q1.shape, fill_value=np.nan, dtype=np.float32)
    valid_indices_1 = np.where(q1 >= 0)  # ensures the square root is taken for positive terms only
    # print('valid_indices_1: ', valid_indices_1)
    temp1[valid_indices_1] = np.sqrt(q1[valid_indices_1])
    w1 = (temp1 ** (1/4)) * ((wavelength / (index_crystal * np.pi)) ** (1 / 2))

    temp2 = np.full(shape=q2.shape, fill_value=np.nan, dtype=np.float32)
    valid_indices_2 = np.where(q2 >= 0)  # ensures the square root is taken for positive terms only
    # print('valid_indices_2: ', valid_indices_2)
    temp2[valid_indices_2] = np.sqrt(q2[valid_indices_2])
    w2 = (temp2 ** (1 / 4)) * ((wavelength / (index_crystal * np.pi)) ** (1 / 2))

    return q1, q2, w1, w2, valid_indices_1, valid_indices_2


def rayleigh_range(waist, wavelength, refraction_index):
    """

    :param waist:
    :param wavelength:
    :param refraction_index:
    :return:
    """
    return np.pi * refraction_index * (waist ** 2) / wavelength


# -- Kaertner classnotes -- #
def waist_mirror1(R1, R2, L, wavelength):
    return (((wavelength * R1) / np.pi) ** 2 * ((R2 - L) / (R1 - L)) * (L / (R1 + R2 - L))) ** (1 / 4)


def waist_mirror3(R1, R2, L, wavelength):
    return (((wavelength * R2) / np.pi) ** 2 * ((R1 - L) / (R2 - L)) * (L / (R1 + R2 - L))) ** (1 / 4)


def waist_intracavity(R1, R2, L, wavelength):
    return ((wavelength / np.pi) ** 2 * (L * (R1 - L) * (R2 - L) * (R1 + R2 - L) / ((R1 + R2 - 2 * L) ** 2))) ** (1 / 4)


# -- Laurat / Boyd -- #
def effective_length(L, l, refractive_index):
    """
    Calculates the effective length of a cavity with a non-linear crystal of length l and index n
    :param L:
    :param l:
    :param refractive_index:
    :return:
    """
    return L - l * (1 - 1/refractive_index)


def effective_Rayleigh_length(L, l, refractive_index, R):
    """
    New Rayleigh length for OPO
    :param L:
    :param l:
    :param refractive_index:
    :param R:
    :return:
    """
    eff_length = effective_length(L, l, refractive_index)
    return np.sqrt(eff_length * (R - eff_length))

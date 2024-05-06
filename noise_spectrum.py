import numpy as np

from utils.settings import settings


def noise_spectrum_x(omega, omega_c, escape_efficiency, epsilon):
    """
    :param omega: Frequency (Hz)
    :param omega_c: Bandwidth
    :param escape_efficiency:
    :param epsilon:

    :return: Density of noise for quadrature x normalized to the vacuum noise.
    """
    return 1 + escape_efficiency * (4 * epsilon) / ((1 - epsilon)**2 + (omega / omega_c)**2)


def noise_spectrum_p(omega, omega_c, escape_efficiency, epsilon):
    """
    :param omega: Frequency (Hz)
    :param omega_c: Bandwidth
    :param escape_efficiency:
    :param epsilon:

    :return: Density of noise for quadrature p normalized to the vacuum noise.
    """
    return 1 - escape_efficiency * (4 * epsilon) / ((1 + epsilon) ** 2 + (omega / omega_c) ** 2)


def negativity(noise_x, noise_p):
    return -((1/noise_x + 1/noise_p - 2)/(noise_x + noise_p - 2)) * ((noise_x + noise_p + 2)/(1/noise_x + 1/noise_p + 2)) * (2/(noise_x + noise_p))

import numpy as np

from utils.settings import settings


def noise_spectrum_x(omega, escape_efficiency, epsilon):
    """
    :param omega: Frequency (Hz)
    :param escape_efficiency:
    :param epsilon:

    :return: Density of noise for quadrature x normalized to the vacuum noise.
    """
    return 1 + escape_efficiency * (4 * epsilon) / ((1 - epsilon)**2 + (omega / settings.omega_c)**2)


def noise_spectrum_p(omega, escape_efficiency, epsilon):
    """
    :param omega: Frequency (Hz)
    :param escape_efficiency:
    :param epsilon:

    :return: Density of noise for quadrature p normalized to the vacuum noise.
    """
    return 1 - escape_efficiency * (4 * epsilon) / ((1 + epsilon) ** 2 + (omega / settings.omega_c) ** 2)

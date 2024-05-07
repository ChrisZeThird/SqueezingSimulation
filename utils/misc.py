from numpy import pi
from utils.settings import settings


def wavelength_to_omega(wavelength):
    return 2 * pi * settings.c / wavelength

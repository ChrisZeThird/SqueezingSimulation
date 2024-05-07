from numpy import pi
from utils.settings import settings


def wavelength_to_omega(wavelength):
    print(settings.c)
    return 2 * pi * settings.c / wavelength

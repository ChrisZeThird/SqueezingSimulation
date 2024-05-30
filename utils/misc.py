import json
import matplotlib.pyplot as plt
from numpy import pi, sin
import numpy as np

from utils.settings import settings


# Wavelength conversion methods
def wavelength_to_omega(wavelength):
    """
    Convert wavelength to frequency
    :param wavelength:
    :return: omega
    """
    return 2 * pi * settings.c / wavelength


def sinc(x):
    """ Redefinition of the sinc function. """
    return sin(x) / x


def kvector(wl, index):
    """ Calculates a wave-vector from wavelength and refractive index.

    :returns: the calculated wave-vector
    """
    wavevector = 2 * pi * index(wl) / wl
    return wavevector


# Importing json
def list_available_crystals(filename):
    with open(filename, "r") as file:
        data = json.load(file)
        crystal_names = list(data.keys())
        return crystal_names


def load_crystal_coefficients(filename, crystal_name):
    with open(filename, "r") as file:
        data = json.load(file)
        coefficients = data.get(crystal_name.upper())
        if coefficients is None:
            available_crystals = list(data.keys())
            raise ValueError(f"Crystal '{crystal_name}' not found in the JSON file. Available crystals: {available_crystals}")
        return coefficients


# Approximate number to the next tenth
def approximate_to_next_ten(number):
    return round(number, -1)

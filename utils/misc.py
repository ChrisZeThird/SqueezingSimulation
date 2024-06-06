import json
import yaml

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


# Importing json/yaml
def list_available_crystals(filename):
    with open(filename, "r") as file:
        data = json.load(file)
        crystal_names = list(data.keys())
        return crystal_names


def load_coefficients(file_path='coefficients.yaml'):
    with open(file_path, 'r') as file:
        coefficients = yaml.safe_load(file)
    return coefficients['coefficients']


# Approximate number to the next tenth
def approximate_to_next_ten(number):
    return round(number, -1)


# Create tuples of string list from two arrays
def arrays_to_tuples_list(arr1, arr2):
    n = len(arr1)
    tuple_list = []
    for i in range(n):
        tuple_list.append((str(arr1[i]), str(arr2[i])))


def convert_np_to_string(arr):
    return np.char.mod('%d', arr)

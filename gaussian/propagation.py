import numpy as np


def rayleigh_range(w, wavelength):
    """
    Calculates the rayleigh range
    :param w:
    :param wavelength:
    :return:
    """
    return np.pi * (w ** 2) / wavelength


def q_parameter(z, w0, wavelength):
    """
    q-parameter according to Svelto
    :param z:
    :param w0:
    :param wavelength:
    :return:
    """
    return z + 1j * rayleigh_range(w0, wavelength)


def propagate(q, prop_matrix):
    """
    q-parameter after propagation
    :param q:
    :param prop_matrix:
    :return:
    """
    A, B, C, D = prop_matrix[0, 0], prop_matrix[0, 1], prop_matrix[1, 0], prop_matrix[1, 1]
    return (A * q + B)/(C * q + D)


# def waist(q, wavelength):
#     """
#     Calculate the waist as a given location for a set q-parameter after propagation
#     :param q:
#     :param wavelength:
#     :return:
#     """
#     return np.sqrt(q.imag * wavelength / np.pi)

def waist(wavelength, prop_matrix, n=1.):
    """
    Waist expression from Tamagawa
    :param wavelength:
    :param n:
    :param prop_matrix:
    :return:
    """
    A, B, C, D = prop_matrix[0, 0], prop_matrix[0, 1], prop_matrix[1, 0], prop_matrix[1, 1]
    print('Stability condition: ', (A + D) / 2)
    z1 = np.sqrt(- B * D / (A * C))
    w1 = np.sqrt(wavelength * z1 / (n * np.pi))
    w2 = n * w1 / np.sqrt((C * z1)**2 + D**2)

    return w1, w2


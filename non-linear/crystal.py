import numpy as np
import yaml

from utils.settings import settings


def Sellmeier(A, B, C, D, E, wavelength):
    """
    Sellmeier general equation
    :return:
    """
    return A + B/(wavelength**2 - C) + D/(wavelength**2 - E)


def kvector(wavelength, index):
    return 2 * np.pi * index / wavelength


class Crystal(object):
    def __init__(self, crystal='KTP', pump_wavelength=780e-9, pump_polarization="V", signal_polarization="V", idler_polarization="V"):
        self.crystal = crystal.upper()
        self.coefficients = self._load_coefficients('coefficients.yaml')

        self.pump_wavelength = pump_wavelength

        self.pump_polarization = pump_polarization
        self.signal_polarization = signal_polarization
        self.idler_polarization = idler_polarization

        self.nh = None
        self.nv = None

    def _load_coefficients(self, yaml_file):
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        return data['coefficients'].get(self.crystal, {})

    def index(self, wavelength):
        """
        Calculate the refraction index from Sellmeier equations
        :return:
        """
        # Create a dictionary that translates crystal axes to integers;
        # this is the pythonic way of implementing a case structure
        _indices = {"X": 0, "Y": 1, "Z": 2}
        # Assume crystal cut along Z, and propagation along X
        _h_index = "Y"
        _v_index = "Z"

        if self.crystal == 'KTP':
            X = self.coefficients['X']
            Y = self.coefficients['Y']
            Z = self.coefficients['Z']

            nx = np.sqrt(Sellmeier(X[1], X[2], X[3], X[4], X[5], wavelength))
            ny = np.sqrt(Sellmeier(Y[1], Y[2], Y[3], Y[4], Y[5], wavelength))
            nz = np.sqrt(Sellmeier(Z[1], Z[2], Z[3], Z[4], Z[5], wavelength))

            # Associate the h- and v-polarized indices with 'x', 'y',
            # and 'z' according to the encoding from above.
            self.nh = [nx, ny, nz][_indices[_h_index]]
            self.nv = [nx, ny, nz][_indices[_v_index]]

        else:
            if self.crystal == 'LN':
                O = self.coefficients['O']
                E = self.coefficients['E']

                # Mole ratio Li / Nb = 0.946
                no = np.sqrt(Sellmeier(O[1], O[2], O[3], O[4], O[5], wavelength))
                ne = np.sqrt(Sellmeier(E[1], E[2], E[3], E[4], E[5], wavelength))

            elif self.crystal == 'BBO':
                O = self.coefficients['O']
                E = self.coefficients['E']
                no = np.sqrt(Sellmeier(O[1], O[2], O[3], O[4], O[5], wavelength))
                ne = np.sqrt(Sellmeier(E[1], E[2], E[3], E[4], E[5], wavelength))

            # Associate the h- and v-polarized indices with ordinary
            # and extraordinary according to the encoding from above
            self.nh = [no, no, ne][_indices[_h_index]]
            self.nv = [no, no, ne][_indices[_v_index]]

    def grating(self):
        """
        Calculates the poling period required for perfect phasematching
        of the supplied fields.

        :return: Poling period
        """
        kpump = kvector(wavelength=settings.pump_wavelength, index=self.index(settings.pump_wavelength))
        ksignal = kvector(wavelength=settings.signal_wavelength, index=self.index(settings.signal_wavelength))
        kidler = kvector(wavelength=settings.idler_wavelength, index=self.index(settings.idler_wavelength))

        mismatch = kpump - ksignal - kidler

        return 2 * np.pi / mismatch

c = Crystal('LN')

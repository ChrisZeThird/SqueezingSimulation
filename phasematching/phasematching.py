import numpy as np

from utils.misc import kvector, sinc


class Phasematching(object):
    def __init__(self,
                 pump_bandwidth,
                 pump_center,
                 pump_wavelength,
                 signal_wavelength,
                 signal_center,
                 idler_wavelength,
                 idler_center,
                 length,
                 grating=None):
        self.pump_bandwidth = pump_bandwidth
        self.pump_center = pump_center
        self.pump_wavelength = pump_wavelength
        self.signal_wavelength = signal_wavelength
        self.signal_center = signal_center
        self.idler_wavelength = idler_wavelength
        self.idler_center = idler_center

        self.length = length
        self.grating = grating

        self.pump_index = None
        self.signal_index = None
        self.idler_index = None

    def grating(self):
        """
        Calculates the poling period required for perfect phasematching
        of the supplied fields.

        :return:
        """
        kpump = kvector(self.pump_center, self.pump_index)
        ksignal = kvector(self.signal_center, self.signal_index)
        kidler = kvector(self.idler_center, self.idler_index)

        return kpump - ksignal - kidler

    def pump_envelope(self):
        """
        Pump function alpha for a Gaussian line shaped laser

        :return:
        """
        return np.exp(-(self.signal_wavelength + self.idler_wavelength - self.pump_center)**2 / (2 * self.pump_bandwidth**2))

    def phase_matching_function(self):
        """
        Calculate the phase mismatch for SPDC process

        :return:
        """
        kpump = kvector(self.pump_wavelength, self.pump_index)
        ksignal = kvector(self.signal_wavelength, self.signal_index)
        kidler = kvector(self.idler_wavelength, self.idler_index)

        return kpump - ksignal - kidler - 2 * np.pi / self.grating()


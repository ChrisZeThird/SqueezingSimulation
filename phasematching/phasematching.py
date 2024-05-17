import numpy as np

from utils.logger import logger
from utils.misc import kvector, sinc
from utils.settings import settings


class Phasematching(object):
    def __init__(self,
                 pump_center,
                 signal_wavelength,
                 signal_center,
                 idler_wavelength,
                 idler_center,
                 length,
                 grating=None):
        self.grating = grating
        self.length = length

        self.pump_index = None
        self.signal_index = None
        self.idler_index = None

        self.pump_center = pump_center
        self.signal_center = signal_center
        self.idler_center = idler_center

        self.signal_wavelength = signal_wavelength
        self.idler_wavelength = idler_wavelength

        self.mismatch = None
        self.pm_function = None

    def polling(self):
        """
        Calculates the poling period required for perfect phasematching
        of the supplied fields.

        :return: Optimal poling of the crystal
        """
        kpump = kvector(self.pump_center, self.pump_index)
        ksignal = kvector(self.signal_center, self.signal_index)
        kidler = kvector(self.idler_center, self.idler_index)

        return kpump - ksignal - kidler

    def wavevector_mismatch(self):
        """
        Calculates the wavevector mismatch matrix.

        :return: Matrix containing the wavevector mismatch in signal and idler frequency plane
        """
        pump_wavelength = 1.0 / (1.0 / self.signal_wavelength + 1.0 / self.idler_wavelength)

        kpump_wl = kvector(pump_wavelength, self.pump_index)
        ksignal_wl = kvector(self.signal_wavelength, self.signal_index)
        kidler_wl = kvector(self.idler_wavelength, self.idler_index)

        return kpump_wl - ksignal_wl - kidler_wl - 2 * np.pi / self.grating

    def phase_matching(self):
        """
        Calculate the phase matching matrix for SPDC

        :return:
        """
        if self.grating is None:
            self.grating = self.polling()

        self.mismatch = self.wavevector_mismatch()

        self.pm_function = sinc(0.5 * self.length * self.mismatch)


class ParametricDownConversion(object):
    def __init__(self):
        self.pm_function = None

        self.signal = np.linspace(start=settings.signal_start, stop=settings.signal_stop, num=settings.signal_steps)
        self.idler = np.linspace(start=settings.idler_start, stop=settings.idler_stop, num=settings.idler_steps)

        self.signal_wavelength, self.idler_wavelength = np.meshgrid(self.signal, self.idler)

# TODO This part is not complete, need to check how it is done in Paderborn program

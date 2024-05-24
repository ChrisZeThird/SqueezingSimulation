import argparse
from dataclasses import asdict, dataclass
from typing import Any

import configargparse
from numpy import inf
from numpy.distutils.misc_util import is_sequence

from utils.logger import logger


@dataclass(init=False, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Settings:
    """
    Storing all settings for this program with default values.
    Setting are loaded from (last override first):
        - default values (in this file)
        - local file (default path: ./settings.yaml)
        - environment variables
        - arguments of the command line (with "--" in front)
    """

    array_points: int = 1000  # size of arrays

    # -- Universal constants -- #
    c: float = 3e8  # speed of light (m/s)

    # -- Cavity characteristics -- #
    omega_c: float = 65000000.0            # bandwidth of the cavity
    transmission_coeff: float = 0.9     # transmission coefficient of the mirrors
    loss_coeff: float = 0.0               # fictitious beam-splitter coefficient
    escape_efficiency: float = 1.0        # proba that if 1 photon escapes it's by output coupler (not due to loss)
    threshold: float = 1.0              # denotes epsilon

    # -- Laser characteristics -- #
    window: float = 50e-9

    input_wavelength: float = 780.0e-9
    lambda_central: float = 780E-9  # signal central wavelength in [m]

    pump_center: float = lambda_central / 2  # pump central wavelength in [m]
    pump_width: float = 1E-9  # pump intensity FWHM in [m]
    pump_temporal_mode: int = 0  # temporal mode order of the pump

    signal_center: float = lambda_central  # signal central wavelength in [m]
    signal_start: float = lambda_central - window  # start of signal plot range in [m]
    signal_stop: float = lambda_central + window  # end of signal plot range in [m]

    idler_start: float = lambda_central - window  # start of idler plot range in [m]
    idler_stop: float = lambda_central + window  # end of idler plot range in [m]

    signal_wavelength: Any = None
    idler_wavelength: Any = None

    signal_steps: int = 800  # points along the signal axis
    idler_steps: int = 800  # points along the idler axis
    rebin_factor: int = 1

    # -- Crystal -- #
    crystal: str = 'PPKTP'
    length: int = 10e-3
    width: float = 3e-6
    height: float = 3e-6
    grating: type(float('inf')) = inf
    temperature: float = 25.0
    pump_index: Any = None
    signal_index: Any = None
    idler_index: Any = None

    hfile: Any = None
    vfile: Any = None

    # -- Polarization -- #
    polarization_type: int = 0
    pump_polarization: str = "V"
    signal_polarization: str = "V"
    idler_polarization: str = "V"

    def __init__(self):
        """
        Create the setting object.
        """
        self._load_file_and_cmd()

    def validate(self):
        """
        Validate settings.
        :return:
        """

        # Possible polarization
        valid_polarizations = ['V', 'H']
        assert self.pump_polarization in valid_polarizations, f'Invalid Pump Polarization: "{self.pump_polarization}"'
        assert self.idler_polarization in valid_polarizations, f'Invalid Idler Polarization: "{self.idler_polarization}"'
        assert self.signal_polarization in valid_polarizations, f'Invalid Signal Polarization: "{self.signal_polarization}"'

        valid_polarization_type = [0, 1, 2]
        assert self.polarization_type in valid_polarization_type
        if self.polarization_type == 0:
            if self.pump_polarization != self.signal_polarization or self.pump_polarization != self.idler_polarization:
                raise ValueError("For Type 0, pump, signal, and idler polarizations must be equal.")
        elif self.polarization_type == 1:
            if self.signal_polarization != self.idler_polarization:
                raise ValueError("For Type 1, signal and idler polarizations must be equal.")
            if self.pump_polarization == self.signal_polarization or self.pump_polarization == self.idler_polarization:
                raise ValueError("For Type 1, pump polarization must be different from signal and idler.")
        elif self.polarization_type == 2:
            if self.signal_polarization == self.idler_polarization:
                raise ValueError("For Type 2, signal and idler polarizations must be different.")
            if self.signal_polarization != self.pump_polarization and self.idler_polarization != self.pump_polarization:
                raise ValueError("For Type 2, pump polarization must match either signal or idler polarization.")

        valid_crystals = ['LN', 'PPLN', 'KTP', 'PPKTP', 'BBO']
        assert self.crystal.upper() in valid_crystals, f"Invalid Crystal name: {self.crystal}"

    def _load_file_and_cmd(self) -> None:
        """
        Load settings from local file and arguments of the command line.
        """
        def str_to_bool(arg_value: str) -> bool:
            """
            Used to handle boolean settings.
            If not the 'bool' type convert all not empty string as true.

            :param arg_value: The boolean value as a string.
            :return: The value parsed as a string.
            """
            if isinstance(arg_value, bool):
                return arg_value
            if arg_value.lower() in {'false', 'f', '0', 'no', 'n'}:
                return False
            elif arg_value.lower() in {'true', 't', '1', 'yes', 'y'}:
                return True
            raise argparse.ArgumentTypeError(f'{arg_value} is not a valid boolean value')

        def type_mapping(arg_value):
            if type(arg_value) == bool:
                return str_to_bool
            if is_sequence(arg_value):
                if len(arg_value) == 0:
                    return str
                else:
                    return type_mapping(arg_value[0])
            if type(arg_value) == float:
                return float
            if type(arg_value) == int:
                return int

        p = configargparse.get_argument_parser(default_config_files=['./settings.yaml'])

        # Spacial argument
        p.add_argument('-s', '--settings', required=False, is_config_file=True,
                       help='path to custom configuration file')

        # Create argument for each attribute of this class
        for name, value in asdict(self).items():
            p.add_argument(f'--{name.replace("_", "-")}',
                           f'--{name}',
                           dest=name,
                           required=False,
                           action='append' if is_sequence(value) else 'store',
                           type=type_mapping(value))

        # Load arguments from file, environment and command line to override the defaults
        for name, value in vars(p.parse_args()).items():
            if name == 'settings':
                continue
            if value is not None:
                # Directly set the value to bypass the "__setattr__" function
                self.__dict__[name] = value

        self.validate()

    def __setattr__(self, name, value) -> None:
        """
        Set an attribute and valid the new value.

        :param name: The name of the attribute
        :param value: The value of the attribute
        """
        if name not in self.__dict__ or self.__dict__[name] != value:
            logger.debug(f'Setting "{name}" changed from "{getattr(self, name)}" to "{value}".')
            self.__dict__[name] = value

    def __delattr__(self, name):
        raise AttributeError('Removing a setting is forbidden for the sake of consistency.')

    def __str__(self) -> str:
        """
        :return: Human-readable description of the settings.
        """
        return 'Settings:\n\t' + \
               '\n\t'.join([f'{name}: {str(value)}' for name, value in asdict(self).items()])


# Singleton setting object
settings = Settings()

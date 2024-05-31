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
    # -- Plot parameter -- #
    number_points: int = 1000   # size of arrays
    cmap_name: str = 'rainbow'  # cmap to use
    alpha: float = 0.3          # alpha channel value for bandwidth highlight

    plot_bandwidth: bool = False    # plot bandwidth region
    plot_waist: bool = False        # plot waist for bow-tie
    plot_kaertner: bool = False     # waist from Kaertner classnotes

    # -- Universal constants -- #
    c: float = 3e8  # speed of light (m/s)

    # -- Cavity characteristics -- #
    fixed_length = 600e-3  # reference length value
    min_L: float = 200e-3
    max_L: float = 100e-3

    omega_c: float = 65000000.0             # bandwidth of the cavity
    transmission_coeff: float = 0.9         # transmission coefficient of the mirrors
    cavity_loss: float = 0.004              # fictitious beam-splitter coefficient
    escape_efficiency: float = 1.0          # proba that if 1 photon escapes it's by output coupler (not due to loss)
    threshold: float = 1.0                  # denotes epsilon

    # -- Transmission coefficients -- #
    min_T: float = 0.1
    max_T: float = 0.3
    R: float = 50e-3    # reference reflection coefficient value
    R1: float = 10e-2   # for Kaertner plot
    R2: float = 11e-2   # for Kaertner plot

    # -- Laser characteristics -- #
    wavelength: float = 780.0e-9

    # -- Bandwidth -- #
    central_freq: float = 6.0e6
    range_freq: float = 0.1  # 10% around central frequency

    # -- Crystal -- #
    crystal_length: int = 10e-3
    crystal_index: float = 1.8396  # PPKTP refraction index
    width: float = 3e-6
    height: float = 3e-6
    grating: type(float('inf')) = inf
    temperature: float = 25.0

    def __init__(self):
        """
        Create the setting object.
        """
        self._load_file_and_cmd()

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

        # self.validate()

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

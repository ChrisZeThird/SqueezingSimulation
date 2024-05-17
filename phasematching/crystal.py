from numpy import sqrt, loadtxt, asarray, cos, sin

from utils.misc import load_crystal_coefficients
from utils.settings import settings

crystal_data = "./data/crystals.json"


class Crystal(object):
    def __init__(self):
        pass

    # def validate_process(self):
    #     """
    #     Ensures that the process is allowed
    #
    #     :return:
    #     """

    def sellmeier(self):
        """
        Calculate the Sellmeier equations.
        - Lithium niobate: taken from Edwards & Lawrence and from Jundt
        - Potassium titanium phosphate: taken from Takaoka et al.

        :return: (nh, nv), tuple containing the Sellmeier equations
        """
        if settings.crystal.upper() in ['LN', 'PPLN']:
            coefficients = load_crystal_coefficients(filename=crystal_data, crystal_name='LN_PPLN')
            T0 = (settings.temperature - 24.5) * (settings.temperature + 570.5)
            TE = (settings.temperature - 24.5) * (settings.temperature + 570.82)


        elif settings.crystal.upper() in ['KTP', 'PPKTP']:
            coefficients = load_crystal_coefficients(filename=crystal_data, crystal_name='KTP_PPKTP')


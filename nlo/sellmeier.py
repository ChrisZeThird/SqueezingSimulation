from numpy import sqrt


def n_z(lambda_um):
    """
    Calculate n_z^2 based on the equation:
    n_z^2 = A + B / (1 - (C * lambda)^2) - D * lambda^2

    Parameters:
        lambda_um (float): Wavelength in micrometers

    Returns:
        float: n_z^2
    """
    A = 2.3136
    B = 1.00012
    C = 0.23831
    D = 0.01679

    square = A + B / (1 - (C / lambda_um)**2) - D * lambda_um**2

    return sqrt(square)


def poling_period(lambda_fund_um):
    """
    Compute the poling period l_poling (in micrometers) for quasi-phase matching in KTP.

    Parameters:
        lambda_fund_um (float): Fundamental wavelength (e.g., 1.56 for 1560 nm)

    Returns:
        float: Poling period in micrometers
    """
    lambda_sh_um = lambda_fund_um / 2  # SHG wavelength
    n_fund = n_z(lambda_fund_um)
    n_sh = n_z(lambda_sh_um)

    delta_n = n_sh - n_fund
    l_poling = lambda_sh_um / delta_n  # formula derived above

    return l_poling

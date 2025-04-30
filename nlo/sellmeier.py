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

import numpy as np


def free_space(d):
    """
    Free space propagation matrix.
    :param d: Distance in meters
    :return: ABCD matrix for free space propagation
    """
    return np.array([[1, d], [0, 1]])


def lens(f):
    """
    Thin lens matrix.
    :param f: Focal length in meters
    :return: ABCD matrix for a thin lens
    """
    return np.array([[1, 0], [-1 / f, 1]])


def curved_mirror(R):
    """
    Curved mirror matrix.
    :param R: Radius of curvature in meters
    :return: ABCD matrix for a curved mirror
    """
    return np.array([[1, 0], [-2 / R, 1]])


def crystal(length, index, f_th=np.infty):
    """
    Crystal matrix.
    :param length: Length of the crystal in meters
    :param index: Index of refraction of the crystal
    :param f_th: Thermal focal length (default is infinity)
    :return: ABCD matrix for a crystal
    """
    return free_space(length/(2 * index)) @ lens(f_th) @ free_space(length/(2 * index))


# # Compute ABCD matrices for free space propagation
# free_space_matrices = np.array([free_space(d) for d in d_c])
#
# # Example: Compute ABCD matrices for a lens
# lens_matrix = lens(f)  # This is fixed, so no broadcasting needed
#
# # Example: Combine matrices (e.g., free space followed by lens) ARRAY BROADCASTING
# combined_matrices = np.array([fs @ lens_matrix for fs in free_space_matrices])
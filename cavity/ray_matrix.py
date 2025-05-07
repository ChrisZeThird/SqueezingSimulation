import numpy as np


def free_space(d):
    """
    Returns ABCD matrices for free-space propagation over distance(s) d.
    Supports scalar or 1D numpy arrays.
    Output shape: (N, 2, 2)
    """
    d = np.atleast_1d(d)
    I = np.ones_like(d)
    Z = np.zeros_like(d)
    return np.stack([
        [I, d],
        [Z, I]
    ], axis=-1).transpose(2, 0, 1)


def curved_mirror(R):
    """
    ABCD matrix for reflection on a curved mirror with radius R.
    Scalar or array input.
    Output shape: (N, 2, 2)
    """
    R = np.atleast_1d(R)
    I = np.ones_like(R)
    Z = np.zeros_like(R)
    minus2_over_R = -2 / R
    return np.stack([
        [I, Z],
        [minus2_over_R, I]
    ], axis=-1).transpose(2, 0, 1)


def refraction(n1, n2):
    """
    Refraction ABCD matrix from index n1 to n2.
    Supports scalar or array input.
    Output shape: (N, 2, 2)
    """
    n1 = np.atleast_1d(n1)
    n2 = np.atleast_1d(n2)
    I = np.ones_like(n1)
    Z = np.zeros_like(n1)
    ratio = n1 / n2
    return np.stack([
        [I, Z],
        [Z, ratio]
    ], axis=-1).transpose(2, 0, 1)


def bowtie_total_matrix(L, d_curved, R, l_crystal, index_crystal):
    """
    Computes the total ABCD matrix of a symmetric bow-tie ring cavity.

    Parameters:
    - L: total round-trip cavity length
    - d_curved: distance between curved mirrors
    - R: radius of curvature of curved mirrors
    - l_crystal: length of nonlinear crystal
    - settings: object with .index_crystal (refractive index of the crystal)

    Returns:
    - A, B, C, D: elements of total ABCD matrix (2x2)
    """
    # Distances
    d1 = l_crystal / 2
    d2 = (d_curved - l_crystal) / 2
    d3 = (L - d_curved) / 2

    M1 = (
        free_space(d3) @
        curved_mirror(R) @
        free_space(d2) @
        refraction(n1=index_crystal, n2=1) @
        free_space(d1)
    )

    A1, B1, C1, D1 = M1.flatten()
    return A1, B1, C1, D1



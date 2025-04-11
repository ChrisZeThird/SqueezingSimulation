import numpy as np


def M_focal(f):
    """
    Propagation matrix through a lens of focal f
    :param f:
    :return: 2x2 matrix
    """
    return np.array([[1, 0],
                     [-1/f, 1]])


def M_free_space(d, n=1.0):
    """
    Propagation matrix in free space over distance d and in a medium of index n
    :param d:
    :param n:
    :return: 2x2 matrix
    """
    return np.array([[1, d/n],
                     [0, 1]])


# Cavity propagation matrices
def M1(L, d_curved, R, l_crystal, index_crystal):
    """
    Propagation matrix starting in the middle of the crystal
    :param L:
    :param d_curved:
    :param R:
    :param l_crystal:
    :param index_crystal:
    :return:
    """
    m1 = M_free_space(d=l_crystal/2, n=index_crystal)
    m2 = M_free_space(d=(d_curved - l_crystal)/2, n=1.)
    m3 = M_focal(f=R/2)
    m4 = M_free_space(d=L - d_curved)

    m_total = m1 @ m2 @ m3 @ m4 @ m3 @ m2 @ m1

    return m_total


def M2(L, d_curved, R, l_crystal, index_crystal):
    """
    Propagation matrix starting in the middle of the flat mirrors
    :param L:
    :param d_curved:
    :param R:
    :param l_crystal:
    :param index_crystal:
    :return:
    """
    m1 = M_free_space(d=(L - d_curved) / 2, n=1.)
    m2 = M_focal(f=R / 2)
    m3 = M_free_space(d=(d_curved - l_crystal)/2, n=1.)
    m4 = M_free_space(d=l_crystal, n=index_crystal)

    m_total = m1 @ m2 @ m3 @ m4 @ m3 @ m2 @ m1

    return m_total

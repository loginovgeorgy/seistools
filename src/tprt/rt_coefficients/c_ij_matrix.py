import numpy as np


def iso_c_ij(vp, vs, density):
    """Constructs c_ij stiffness matrix for isotropic medium.
    :param vp: P-wave velocity in the medium
    :param vs: S-wave velocity in the medium
    :param density: value of medium's density
    :return: 6x6-matrix of elastic constants
    """

    c_ij = np.zeros((6, 6))  # index "1" beside "c_ij" marks that this is a variable,
    # not a function defined above

    c_ij[0, 0] = density*(vp ** 2)  # lambda +  2 * mu
    c_ij[1, 1] = c_ij[0, 0]
    c_ij[2, 2] = c_ij[0, 0]

    c_ij[3, 3] = density*(vs ** 2)  # mu
    c_ij[4, 4] = c_ij[3, 3]
    c_ij[5, 5] = c_ij[3, 3]

    c_ij[0, 1] = c_ij[0, 0] - 2 * c_ij[3, 3]  # lambda
    c_ij[0, 2] = c_ij[0, 1]
    c_ij[1, 2] = c_ij[0, 1]
    c_ij[1, 0] = c_ij[0, 1]
    c_ij[2, 0] = c_ij[0, 1]
    c_ij[2, 1] = c_ij[0, 1]
    
    return c_ij


def voigt_notation(i, j):
    """Defines transition rule c_ijkl -> c_ij (from fourth-rank stiffness tensor to second-rank matrix using symmetry).

    11 -> 1; 22 -> 2; 33 -> 3;
    23,32 -> 4; 13,31 -> 5; 12,21 -> 6.
    Remember that in Python indexing starts from 0, not from 1.

    :param i: index i from ij pair (index k from kl pair)
    :param j: index j from ij pair (index l from kl pair)
    :return: corresponding index i of 6x6 stiffness matrix
    """

    # if i == j:
    #     return i
    # if [i, j] == [1, 2] or [i, j] == [2, 1]:
    #     return 3
    # if [i, j] == [0, 2] or [i, j] == [2, 0]:
    #     return 4
    # if [i, j] == [0, 1] or [i, j] == [1, 0]:
    #     return

    return i * (i == j) + (6 - i - j) * (i != j)


def c_ijkl_from_c_ij(c_ij):
    """Constructs full stiffness tensor c_ijkl from its matrix form c_ij.

    :param c_ij: stiffness matrix c_ij
    :return: 3x3x3x3-tensor of elastic constants
    """
    
    c_ijkl1 = np.zeros((3, 3, 3, 3))
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    
                    from_i = voigt_notation(i, j)
                    from_j = voigt_notation(k, l)
                    
                    c_ijkl1[i, j, k, l] = c_ij[from_i, from_j]
                    
    return c_ijkl1

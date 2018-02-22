import numpy as np


def _to_vector(m):
    m = m.ravel()
    if len(m) != 9:
        raise ValueError(""" 
        Moment must be matrix of size 3*3: 
        M[11, 12, 13, 
          12, 22, 23, 
          13, 23, 33]. 
        {} elements matrix passed.
        """.format(len(m)))

    vector = m[[0, 1, 2, 4, 5, 8]]
    return vector


def _to_matrix(m):
    m = m.ravel()
    if len(m) != 6:
        raise ValueError(""" 
        Moment must be vector of 6 elements: M[11, 12, 13, 22, 23, 33]. 
        {} elements vector passed.
        """.format(len(m)))

    matrix = np.array([m[[0, 1, 2]],
                       m[[1, 3, 4]],
                       m[[2, 4, 5]]])
    return matrix


CONVERT_DICT = {'matrix': _to_matrix,
                'vector': _to_vector}


def mt_convert(m, d=None):
    m = m.ravel()
    if not dir:
        m = CONVERT_DICT[d]
        return m

    if len(m) == 6:
        m = _to_matrix(m)
    elif len(m) == 9:
        m = _to_vector(m)
    else:
        return 0
    return m

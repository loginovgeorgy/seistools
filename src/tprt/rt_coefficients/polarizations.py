import numpy as np

from .c_ij_matrix import iso_c_ij, voigt_notation, c_ijkl_from_c_ij


def christoffel(c_ij, n):
    """Constructs Christoffel matrix from 6x6 stiffness matrix c_ij and direction vector n.

    :param c_ij: 6x6 stiffness matrix c_ij
    :param n: vector of direction
    :return: Christoffel matrix Gik = c_ijkl n_j n_l where c_ijkl is full stiffness tensor constructed from c_ij matrix
    """

    c_ijkl = c_ijkl_from_c_ij(c_ij)

    # return np.einsum("ijkl, j, l", c_ijkl, n, np.conj(n))  # I am not sure if we really need complex conjugation
    return np.einsum("ijkl, j, l", c_ijkl, n, n)


def polarizations(c_ij, n):
    """Computes possible polarizations for elastic waves propagating in given direction.

    :param c_ij: 6x6 stiffness matrix c_ij
    :param n: vector of direction
    :return: possible wave polarizations for given direction
    """

    g = christoffel(c_ij, n)  # Christoffel matrix for this medium and this direction

    # _, s, v = np.linalg.svd(g)  # s is vector of eigenvalues of g ad v is matrix of its eigenvectors

    s, v = np.linalg.eig(g)
    # s, v = np.linalg.eigh(g)

    # v = np.real(v)  # g is positive-determined matrix but for sake of security we take real parts of eigenvectors
    # s = np.real(s)  # and eigenvalues
    # s = np.abs(s)
    # s = np.sqrt(s)

    idx_sort = np.abs(s).argsort()[::-1]
    v = v[:, idx_sort]
    # v = v[:, np.array([2, 0, 1])]

    if np.dot(n, v[:, 0]) < 0:
        v = -v

    return v


def polarizations_alt(c_ij1, n): # index "1" beside "c_ij" marks that this is a variable,
    # not a function defined in another module

    # задаём формат вывода
    u = np.array(0, dtype=complex)
    u.resize(3, 3)

    # находим весь тензор c_ijkl1:
    c_ijkl1 = c_ijkl_from_c_ij(c_ij1) # the same index with the same purpose

    # Строим матрицу Кристоффеля Гik = c_ijkl1*nj*nl:
    Гik = np.array(0, dtype=complex)
    Гik.resize(3, 3)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    Гik[i, k] = Гik[i, k] + c_ijkl1[i, j, k, l] * n[j] * n[l]

    # И находим собственные векторы этой матрицы. Они и будут векторами поляризации.
    # Однако эти векторы будут в невесть каком порядке. Исправим это: отсортируем соственные числа по возрастанию:
    eigenvalues = np.linalg.eig(Гik)[0]
    I = np.argsort(eigenvalues)  # запоминаем перестановку

    polariz = np.linalg.eig(Гik)[1]  # отсюда будем брать векторы поляризвации.

    # поляризация продольной волны совпадает с направлением распространения волны
    u[:, 0] = polariz[:, I[2]]  # поляризация продольной волны
    if u[:, 0].dot(n) < 0:
        u[:, 0] = - u[:, 0]  # Продольная волна должна быть поляризована в направлении распространения.

    # может получиться так, что все три вектора поляризации будут направлены вдоль осей координат. В этом случае надо
    # директивно  задать "правильные" направления поляризаций.
    # В изотропной среде, с учётом особенностей нашей задачи, такая ситуация может возникнуть только при нормальном
    # падении:

    if abs(u[2, 0]) == 1: # если продольная волна поляризована по Z

        u[:, 1] = np.array([1, 0, 0])
        u[:, 2] = np.array([0, 1, 0])

        return u

    # поляризацию поперечной волны можно задать по-разному, но мы зададим так:

    if np.dot(polariz[:,I[0]], np.array([0, 0, 1])) == 0: # если собственный вектор, соответствующий самому маленькому
        # собственному числу, ортогонален вертикальному вектору, то он соответствует S2 (SH) - волне:

        u[:,1] = polariz[:,I[1]] #поляризация "первой" поперечной волны
        u[:,2] = polariz[:,I[0]] #поляризация "второй" поперечной волны


    else: #если же нет...

        u[:,1] = polariz[:,I[0]] #поляризация "первой" поперечной волны
        u[:,2] = polariz[:,I[1]] #поляризация "второй" поперечной волны

    # Осталось проверить последнее: не получается ли тройка S1, S2 и P левой. Заметим, что поляризация вектора P уже правильная. Будем
    # варьировать знак S2.

    if np.dot(u[:, 0], np.cross(u[:, 1], u[:, 2])) < 0:

        u[:, 2] = - u[:, 2]

    return u
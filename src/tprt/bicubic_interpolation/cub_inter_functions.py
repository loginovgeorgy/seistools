# Источник на применяемую теорию: Марчук Г.И. Методы вычислительной математики. М.: Наука. Главная редакция физико-
# математической литературы, 1980

import numpy as np
from scipy.linalg import solve_banded


def get_left_i(x_set, x_target):
    """

    :param x_set: strictly ascending 1D-numerical array
    :param x_target: target value of x
    :return: number of the nearest x_target's neighbour in x_set
    """

    # Ищет номер ближайшего слева к x элемента из x_set. Т.е. x - число, а x_set - упорядоченный по возрастанию массив
    # чисел.
    
    nearest_i = np.argmin(abs(x_set[: - 1] - x_target))
    
    if nearest_i == 0: # this condition provides that all x_target < x_set[0] will be bound to x[0], not to x[- 1]
        
        return nearest_i
    
    if x_target < x_set[nearest_i]: # this condition provides that all x_target inside x_set will be bound to their left
        # neighbour
        
        return nearest_i - 1
    
    return nearest_i # in that case the nearest neighbour is already to the left from x_target

    # # Граничные случаи:
    # if x_target <= x_set[0]:
    #     return 0
    # if x_target >= x_set[- 1]:
    #     return x_set.shape[0] - 2

    # Переходим к бинарному поиску
    # i = 0
    # j = x_set.shape[0]
    # mid = 0

    # while (i < j):

    #     mid = round((i + j) / 2) # идём на середину

    #     # Проверяем, не попали ли мы уже на нужный нам элемент
    #     if (x_set[mid] == x_target):

    #         return mid

    #     # Если наш элемент меньше, чем срединный, ищем в левой половине.

    #     if x_target < x_set[mid]:

    #         # Cначала проверяем, не является ли наш элемент граничным случаем для этой половины:

    #         if mid > 0 and x_set[mid - 1] < x_target:

    #             return mid - 1

            # Приписываем значение mid правой границе исследуемой области и идём на следующую итерацию:

    #         j = mid

    #     # Если же мы оказались справа от середины:

    #     else:

    #         # Аналогичная проверка на "граничность":

    #         if mid < x_set.shape[0] - 1 and x_target < x_set[mid + 1]:

    #             return mid

    #         # Аналогично, сдвигаем левую границу к mid:

    #         i = mid

    # # В конце цикла останется единственный элемент:

    # return mid

# 1D-interpolation functions:


def second_derivatives(x_set, z_set):
    """

    :param x_set: strictly ascending 1D-numerical array
    :param z_set: 1D/2D-numerical array of corresponding values z = z_j(x) where index j denotes number of "profile"
    :return: 1D/2D-numerical array of second derivatives z'' = z_j''(x) at points from x_set
    """

    # Returns array of second derivatives of z at points x = x_set[i].

    # A system of linear equations is solved:
    # A m = B
    # Matrix A is three-diagonal.

    steps = np.diff(x_set) # steps between x[i] and x[i + 1] for all i

    A = np.zeros((3, x_set.shape[0] - 2)) # here we shall write all three non-zero diagonals of the system's matrix
    B = np.zeros(A.shape[1]) # right part of the system

    A[0, 1:] = steps[1: - 1] / 6
    A[1, :] = (steps[0: - 1] + steps[1:]) / 3
    A[2, 0: - 1] = steps[1: - 1] / 6

    # A[1, 0] = (x_set[2] - x_set[0]) / 3
    # A[2, 0] = (x_set[2] - x_set[1]) / 6

    # A[0, -1] = (x_set[-2] - x_set[-3]) / 6
    # A[1, -1] = (x_set[-1] - x_set[-3]) / 3

    # B[0] = (z_set[2] - z_set[1]) / (x_set[2] - x_set[1]) - (z_set[1] - z_set[0]) / (x_set[1] - x_set[0])

    # B[-1] = (z_set[- 1] - z_set[- 2]) / (x_set[- 1] - x_set[- 2]) -\
    #         (z_set[- 2] - z_set[- 3]) / (x_set[- 2] - x_set[- 3])

    # for i in np.arange(1, A.shape[1] - 1, 1):

    #     A[0, i] = (x_set[i + 1] - x_set[i]) / 6
    #     A[1, i] = (x_set[i + 2] - x_set[i]) / 3
    #     A[2, i] = (x_set[i + 1] - x_set[i]) / 6

    #     B[i] = (z_set[i + 2] - z_set[i + 1]) / (x_set[i + 2] - x_set[i + 1]) -\
    #            (z_set[i + 1] - z_set[i]) / (x_set[i + 1] - x_set[i])

    # Matrix A now corresponds to the system for a single "profile".
    # If this "profile" is really single everything is straightforward:

    if len(z_set.shape) == 1:

        B = z_set[: - 2] / steps[: - 1] - z_set[1: - 1] * (1 / steps[: - 1] + 1 / steps[1:]) + z_set[2 :] / steps[1:]

        sec_deriv = solve_banded((1, 1), A, np.ravel(B)) # here are second derivatives in points x[1]...x[- 2].

        sec_deriv = np.append(0, sec_deriv)
        sec_deriv = np.append(sec_deriv, 0)

    # If there are several of them, system is slightly different:

    else:

        A = np.tile(A, z_set.shape[0])

        B = z_set[:, : - 2] / steps[: - 1] - \
            z_set[:, 1: - 1] * (1 / steps[: - 1] + 1 / steps[1:]) + \
            z_set[:, 2 :] / steps[1:]

        sec_deriv = solve_banded((1, 1), A, np.ravel(B)) # here are second derivatives in points x[1]...x[- 2].
        sec_deriv = np.reshape(sec_deriv, (z_set.shape[0], z_set.shape[1] - 2))

        sec_deriv = np.append(np.zeros((z_set.shape[0], 1)), sec_deriv, axis=1)
        sec_deriv = np.append(sec_deriv, np.zeros((z_set.shape[0], 1)), axis=1)

    return sec_deriv


def one_dim_polynomial(x_set, z_set):
    """

    :param x_set: strictly ascending 1D-numerical array
    :param z_set: 1D/2D-numerical array of corresponding values z = z_j(x) where index j denotes number of "profile"
    :return: array of polynomial coefficients of cubic spline interpolation of z(x) function
    """

    # Returns array of coefficients of the interpolation ploynomials for function z_set defined on grid x_set.

    steps = np.diff(x_set) # steps between x[i] and x[i + 1] for all i

    # If there is a single "profile":
    if len(z_set.shape) == 1:

        one_dim_coefficients = np.zeros((x_set.shape[0] - 1, 4)) # the first index corresponds to the left nearest neighbour
        # and the second index denotes power of x in polynomial

        m_i = second_derivatives(x_set, z_set)

        one_dim_coefficients[:, 0] = m_i[: - 1] * x_set[1:] ** 3 / (6 * steps) - \
                                     m_i[1:] * x_set[: - 1] ** 3 / (6 * steps) + \
                                     (z_set[: - 1] - m_i[: - 1] * steps ** 2 / 6) * x_set[1:] / steps - \
                                     (z_set[1:] - m_i[1:] * steps ** 2 / 6) * x_set[: - 1] / steps

        one_dim_coefficients[:, 1] = - m_i[: - 1] * 3 * x_set[1:] ** 2 / (6 * steps) + \
                                     m_i[1:] * 3 * x_set[: - 1] ** 2 / (6 * steps) - \
                                     (z_set[: - 1] - m_i[: - 1] * steps ** 2 / 6) / steps + \
                                     (z_set[1:] - m_i[1:] * steps ** 2 / 6) / steps

        one_dim_coefficients[:, 2] = m_i[: - 1] * 3 * x_set[1:] / (6 * steps) - \
                                     m_i[1:] * 3 * x_set[: - 1] / (6 * steps)

        one_dim_coefficients[:, 3] = - m_i[: - 1] / (6 * steps) + \
                                     m_i[1:] / (6 * steps)

    # If there are several of them:
    else:

        one_dim_coefficients = np.zeros((z_set.shape[0], x_set.shape[0] - 1, 4)) # the first index corresponds to the
        # "profile" number, the second index corresponds to the left nearest neighbour and the third index denotes power
        # of x in polynomial

        m_i = second_derivatives(x_set, z_set)

        one_dim_coefficients[:, :, 0] = m_i[:, : - 1] * x_set[1:] ** 3 / (6 * steps) - \
                                        m_i[:, 1:] * x_set[: - 1] ** 3 / (6 * steps) + \
                                        (z_set[:, : - 1] - m_i[:, : - 1] * steps ** 2 / 6) * x_set[1:] / steps - \
                                        (z_set[:, 1:] - m_i[:, 1:] * steps ** 2 / 6) * x_set[: - 1] / steps

        one_dim_coefficients[:, :, 1] = - m_i[:, : - 1] * 3 * x_set[1:] ** 2 / (6 * steps) + \
                                        m_i[:, 1:] * 3 * x_set[: - 1] ** 2 / (6 * steps) - \
                                        (z_set[:, : - 1] - m_i[:, : - 1] * steps ** 2 / 6) / steps + \
                                        (z_set[:, 1:] - m_i[:, 1:] * steps ** 2 / 6) / steps

        one_dim_coefficients[:, :, 2] = m_i[:, : - 1] * 3 * x_set[1:] / (6 * steps) - \
                                        m_i[:, 1:] * 3 * x_set[: - 1] / (6 * steps)

        one_dim_coefficients[:, :, 3] = - m_i[:, : - 1] / (6 * steps) + \
                                        m_i[:, 1:] / (6 * steps)

    # for i in np.arange(1, x_set.shape[0], 1):
    #
    #     step_i = x_set[i] - x_set[i - 1]
    #
    #     one_dim_coefficients[i - 1, 0] = m_i[i - 1] * x_set[i] ** 3 / (6 * step_i) - \
    #                                      m_i[i] * x_set[i - 1] ** 3 / (6 * step_i) + \
    #                                      (z_set[i - 1] - m_i[i - 1] * step_i ** 2 / 6) * x_set[i] / step_i - \
    #                                      (z_set[i] - m_i[i] * step_i ** 2 / 6) * x_set[i - 1] / step_i
    #
    #     one_dim_coefficients[i - 1, 1] = - m_i[i - 1] * 3 * x_set[i] ** 2 / (6 * step_i) + \
    #                                      m_i[i] * 3 * x_set[i - 1] ** 2 / (6 * step_i) - \
    #                                      (z_set[i - 1] - m_i[i - 1] * step_i ** 2 / 6) / step_i + \
    #                                      (z_set[i] - m_i[i] * step_i ** 2 / 6) / step_i
    #
    #     one_dim_coefficients[i - 1, 2] = m_i[i - 1] * 3 * x_set[i] / (6 * step_i) - \
    #                                      m_i[i] * 3 * x_set[i - 1] / (6 * step_i)
    #
    #     one_dim_coefficients[i - 1, 3] = - m_i[i - 1] / (6 * step_i) + \
    #                                      m_i[i] / (6 * step_i)

    return one_dim_coefficients

# 2D-interpolation functions


def two_dim_polynomial(x_set, y_set, z_set):
    """

    :param x_set: strictly ascending 1D-numerical array
    :param y_set: strictly ascending 1D-numerical array
    :param z_set: 2D-numerical array of corresponding values z = z(x, y)
    :return: array of polynomial coefficients of bicubic spline interpolation of z(x, y) function
    """

    # Returns array of polynomial coefficients for bicubic interpolation.

    # We'll need several arrays of coefficients that can be found using one-dimensional interpolation:

    # 1. Array of polynoial coefficients for cross-sections z(xi, y):

    xi_coeff = one_dim_polynomial(y_set, z_set)

    # xi_coeff = np.zeros((x_set.shape[0], y_set.shape[0] - 1, 4))

    # 2. Array of polynomial coefficients for cross-section of the second partial derivatives of z with respect to x
    # along lines x = xi (so, this function will be a cubic polynomial of y):

    z_xx = second_derivatives(x_set, np.transpose(z_set))
    z_xx_coeff = one_dim_polynomial(y_set, z_xx)

    # z_xx = np.zeros((x_set.shape[0], y_set.shape[0]))  # array of d2z/dx2 on the grid
    # z_xx_coeff = np.zeros((x_set.shape[0], y_set.shape[0] - 1, 4))  # polynomial coefficients for d2z/dx2

    # Let's fill these arrays:

    # for j in range(y_set.shape[0]):
    #
    #     z_xx[:, j] = second_derivatives(x_set, z_set[:, j])
    #
    # for i in range(x_set.shape[0]):
    #
    #     xi_coeff[i, :, :] = one_dim_polynomial(y_set, z_set[i, :])
    #     z_xx_coeff[i, :, :] = one_dim_polynomial(y_set, z_xx[i, :])

    # Now we have everything what we need. Let's construct new array for two-dimensional interpolation polynomial
    # coefficients:

    two_dim_coefficients = np.zeros((x_set.shape[0] - 1, y_set.shape[0] - 1, 4, 4))

    x_steps_transp = np.transpose(np.array([np.diff(x_set)]))
    x_set_transp = np.transpose(np.array([x_set]))  # the transposition is performed for correct multiplication below

    for m in range(4):

        two_dim_coefficients[:, :, 0, m] = (xi_coeff[: - 1, :, m] * x_set_transp[1:] -
                                            xi_coeff[1:, :, m] * x_set_transp[: - 1]) / x_steps_transp - \
                                           (z_xx_coeff[: - 1, :, m] * x_set_transp[1:] -
                                            z_xx_coeff[1:, :, m] * x_set_transp[: - 1]) * x_steps_transp / 6 + \
                                           (z_xx_coeff[: - 1, :, m] * x_set_transp[1:] ** 3 -
                                            z_xx_coeff[1:, :, m] * x_set_transp[: - 1] ** 3) / x_steps_transp / 6

        two_dim_coefficients[:, :, 1, m] = (xi_coeff[1:, :, m] -
                                            xi_coeff[: - 1, :, m]) / x_steps_transp - \
                                           (z_xx_coeff[1:, :, m] -
                                            z_xx_coeff[: - 1, :, m]) * x_steps_transp / 6 + \
                                           (z_xx_coeff[1:, :, m] * x_set_transp[: - 1] ** 2 -
                                            z_xx_coeff[: - 1, :, m] * x_set_transp[1:] ** 2) / x_steps_transp / 2

        two_dim_coefficients[:, :, 2, m] = (z_xx_coeff[: - 1, :, m] * x_set_transp[1:] -
                                            z_xx_coeff[1:, :, m] * x_set_transp[: - 1]) / x_steps_transp / 2

        two_dim_coefficients[:, :, 3, m] = (z_xx_coeff[1:, :, m] -
                                            z_xx_coeff[: - 1, :, m]) / x_steps_transp / 6

    # two_dim_coefficients = np.zeros((x_set.shape[0] - 1, y_set.shape[0] - 1, 4, 4))
    #
    # # x_steps = np.diff(x_set)
    # #
    # # for m in range(4):
    # #
    # #     two_dim_coefficients[:, :, 0, m] = (xi_coeff[: - 1, :, m] * )
    #
    # for i in np.arange(1, x_set.shape[0], 1):
    #
    #     step_i = x_set[i] - x_set[i - 1]
    #
    #     for j in np.arange(1, y_set.shape[0], 1):
    #
    #         for m in range(4):
    #
    #             two_dim_coefficients[i - 1, j - 1, 0, m] = (xi_coeff[i - 1, j - 1, m] * x_set[i] -
    #                                                         xi_coeff[i, j - 1, m] * x_set[i - 1]) / step_i -\
    #                                                        step_i * (z_xx_coeff[i - 1, j - 1, m] * x_set[i] -
    #                                                                  z_xx_coeff[i, j - 1, m] * x_set[i - 1]) / 6 +\
    #                                                        (z_xx_coeff[i - 1, j - 1, m] * x_set[i] ** 3 -
    #                                                         z_xx_coeff[i, j - 1, m] * x_set[i - 1] ** 3) / step_i / 6
    #
    #             two_dim_coefficients[i - 1, j - 1, 1, m] = (xi_coeff[i, j - 1, m] -
    #                                                         xi_coeff[i - 1, j - 1, m]) / step_i -\
    #                                                        step_i * (z_xx_coeff[i, j - 1, m] -
    #                                                                  z_xx_coeff[i - 1, j - 1, m]) / 6 +\
    #                                                        (z_xx_coeff[i, j - 1, m] * x_set[i - 1] ** 2 -
    #                                                         z_xx_coeff[i - 1, j - 1, m] * x_set[i] ** 2) / step_i / 2
    #
    #             two_dim_coefficients[i - 1, j - 1, 2, m] = (z_xx_coeff[i - 1, j - 1, m] * x_set[i] -
    #                                                         z_xx_coeff[i, j - 1, m] * x_set[i - 1]) / step_i / 2
    #
    #             two_dim_coefficients[i - 1, j - 1, 3, m] = (z_xx_coeff[i, j - 1, m] -
    #                                                         z_xx_coeff[i - 1, j - 1, m]) / step_i / 6

    return two_dim_coefficients


def two_dim_inter(two_dim_coeff, x_set, y_set, xy_target):
    """

    :param two_dim_coeff: array of polynomial coefficients of bicubic spline interpolation of z(x, y) function
    :param x_set: strictly ascending 1D-numerical array
    :param y_set: strictly ascending 1D-numerical array
    :param xy_target: target value of (x, y); xy_target = [x_target, y_target]
    :return: value of interpolated function z(x,y) at the target point
    """

    # Returns value of z(x, y) corresponding to the pre-constructed array of polynomial coefficients and the
    # parametrization grid.

    i = get_left_i(x_set, xy_target[0])
    j = get_left_i(y_set, xy_target[1])

    z = 0 # this is the value of z(x, y). We shall compute it in a cycle below.

    for m in np.arange(0, 4, 1):
        for n in np.arange(0, 4, 1):

            z = z + two_dim_coeff[i, j, m, n] * (xy_target[0] ** m) * (xy_target[1] ** n)

    return z


def two_dim_inter_dx(two_dim_coeff, x_set, y_set, xy_target):
    """

    :param two_dim_coeff: array of polynomial coefficients of bicubic spline interpolation of z(x, y) function
    :param x_set: strictly ascending 1D-numerical array
    :param y_set: strictly ascending 1D-numerical array
    :param xy_target: target value of (x, y); xy_target = [x_target, y_target]
    :return: value of partial derivative d / dx of interpolated function z(x,y) at the target point:
    dz / dx |x = x_target, y = y_target
    """

    # Returns partial derivative dz(x, y) / dx corresponding to the pre-constructed array of polynomial
    # coefficients and the parametrization grid.

    i = get_left_i(x_set, xy_target[0])
    j = get_left_i(y_set, xy_target[1])

    z_x = 0 # this is the value of dz(x, y) / dx. We shall compute it in a cycle below.

    for m in np.arange(1, 4, 1):
        for n in np.arange(0, 4, 1):

            z_x = z_x + two_dim_coeff[i, j, m, n] * (m * xy_target[0] ** (m - 1)) * (xy_target[1] ** n)

    return z_x


def two_dim_inter_dy(two_dim_coeff, x_set, y_set, xy_target):
    """

    :param two_dim_coeff: array of polynomial coefficients of bicubic spline interpolation of z(x, y) function
    :param x_set: strictly ascending 1D-numerical array
    :param y_set: strictly ascending 1D-numerical array
    :param xy_target: target value of (x, y); xy_target = [x_target, y_target]
    :return: value of partial derivative d / dy of interpolated function z(x,y) at the target point:
    dz / dy |x = x_target, y = y_target
    """

    # Returns partial derivative dz(x, y) / dy corresponding to the pre-constructed array of polynomial
    # coefficients and the parametrization grid.

    i = get_left_i(x_set, xy_target[0])
    j = get_left_i(y_set, xy_target[1])

    z_y = 0 # this is the value of dz(x, y) / dy. We shall compute it in a cycle below.

    for m in np.arange(0, 4, 1):
        for n in np.arange(1, 4, 1):

            z_y = z_y + two_dim_coeff[i, j, m, n] * (xy_target[0] ** m) * (n * xy_target[1] ** (n - 1))

    return z_y


def two_dim_inter_dx_dx(two_dim_coeff, x_set, y_set, xy_target):
    """

    :param two_dim_coeff: array of polynomial coefficients of bicubic spline interpolation of z(x, y) function
    :param x_set: strictly ascending 1D-numerical array
    :param y_set: strictly ascending 1D-numerical array
    :param xy_target: target value of (x, y); xy_target = [x_target, y_target]
    :return: value of partial derivative d^2 / dx^2 of interpolated function z(x,y) at the target point:
    d^2 z / dx^2 |x = x_target, y = y_target
    """

    # Returns partial derivative d^2z(x, y) / dx^2 corresponding to the pre-constructed array of polynomial
    # coefficients and the parametrization grid.

    i = get_left_i(x_set, xy_target[0])
    j = get_left_i(y_set, xy_target[1])

    z_xx = 0 # this is the value of dz(x, y) / dy. We shall compute it in a cycle below.

    for m in np.arange(2, 4, 1):
        for n in np.arange(0, 4, 1):

            z_xx = z_xx + two_dim_coeff[i, j, m, n] * (m * (m - 1) * xy_target[0] ** (m - 2)) * (xy_target[1] ** n)

    return z_xx


def two_dim_inter_dy_dy(two_dim_coeff, x_set, y_set, xy_target):
    """

    :param two_dim_coeff: array of polynomial coefficients of bicubic spline interpolation of z(x, y) function
    :param x_set: strictly ascending 1D-numerical array
    :param y_set: strictly ascending 1D-numerical array
    :param xy_target: target value of (x, y); xy_target = [x_target, y_target]
    :return: value of partial derivative d^2 / dy^2 of interpolated function z(x,y) at the target point:
    d^2 z / dy^2 |x = x_target, y = y_target
    """

    # Returns partial derivative d^2z(x, y) / dy^2 corresponding to the pre-constructed array of polynomial
    # coefficients and the parametrization grid.

    i = get_left_i(x_set, xy_target[0])
    j = get_left_i(y_set, xy_target[1])

    z_yy = 0 # this is the value of dz(x, y) / dy. We shall compute it in a cycle below.

    for m in np.arange(0, 4, 1):
        for n in np.arange(2, 4, 1):

            z_yy = z_yy + two_dim_coeff[i, j, m, n] * (xy_target[0] ** m) * (n * (n - 1) * xy_target[1] ** (n - 2))

    return z_yy


def two_dim_inter_dx_dy(two_dim_coeff, x_set, y_set, xy_target):
    """

    :param two_dim_coeff: array of polynomial coefficients of bicubic spline interpolation of z(x, y) function
    :param x_set: strictly ascending 1D-numerical array
    :param y_set: strictly ascending 1D-numerical array
    :param xy_target: target value of (x, y); xy_target = [x_target, y_target]
    :return: value of partial derivative d^2 / dx dy of interpolated function z(x,y) at the target point:
    d^2 z / dx dy |x = x_target, y = y_target
    """

    # Returns partial derivative d^2z(x, y) / dx dy corresponding to the pre-constructed array of polynomial
    # coefficients and the parametrization grid.

    i = get_left_i(x_set, xy_target[0])
    j = get_left_i(y_set, xy_target[1])

    z_y = 0 # this is the value of dz(x, y) / dy. We shall compute it in a cycle below.

    for m in np.arange(1, 4, 1):
        for n in np.arange(1, 4, 1):

            z_y = z_y + two_dim_coeff[i, j, m, n] * (m * xy_target[0] ** (m - 1)) * (n * xy_target[1] ** (n - 1))

    return z_y

# Потребуется минимизация, и в силу особенностей синтаксиса надо определить ещё одну функцию. За подробностями - в
# классе "GridHorizon".


def difference(s, x_set, y_set, two_dim_coeff, sou, vec):
    """

    :param s: length's value
    :param x_set: strictly ascending 1D-numerical array
    :param y_set: strictly ascending 1D-numerical array
    :param two_dim_coeff: array of polynomial coefficients of bicubic spline interpolation of z(x, y) function
    :param sou: coordinates of starting point
    :param vec: direction vector
    :return: distance along z axis from the point sou + s * vec and to the surface defined by the interpolation
    polynomial
    """

    return abs(two_dim_inter(two_dim_coeff, x_set, y_set, np.array([sou[0:2] + s * vec[0:2]])) - \
               (sou[2] + s * vec[2]))


def parabola(points, x_target):
    """

    :param points: coordinates of three points in form points = [point_1, point_2, point_3], where  point_i = [x_i, z_i]
    :param x_target: target value of x
    :return: value of parabola z = A * x**2 + B * x + C which fits all three points at x = x_target
    """
    # Строит параболу, проходящую через три точки points = [point_1, point_2, point_3], где все point_i = [x_i, z_i], и
    # возвращает значение этой параболы в точке aim_x.
    # Парабола ищется в виде z = A * x**2 + B * x + C

    Sys = np.zeros((3, 3)) # матрица системы для поиска коэффициентов параболы
    Rpart = np.zeros(3) # правая часть системы

    for i in range(3):

        Sys[i, 0] = points[i][0] ** 2 # x**2
        Sys[i, 1] = points[i][0] # x
        Sys[i, 2] = 1 # свободный член

        Rpart[i] = points[i][1] # z

    A, B, C = np.linalg.solve(Sys, Rpart)

    return A * x_target**2 + B * x_target + C


def one_dim_parab_inter(x_set, z_set, x_target):
    """

    :param x_set: strictly ascending 1D-numerical array
    :param z_set: 1D-numerical array of corresponding values z = z(x)
    :param x_target: 1D-numerical array of corresponding values z = z(x)
    :return: average value of two neighbour parabolas at x = x_target
    """
    # По заданной сетке и дискретно заданной функции func = np.array([f1, f2, ..., fn]) строит одномерную
    # усреднённо-параболическую интерполяцию в точке с координатой aim_x. Усреднённо-параболическая интерполяция
    # состоит в следующем: пусть есть три точки (x1, y1), (x2, y2) и (x3, y3). Через них проводится парабола. Далее
    # смешаемся на одну точку вправо: теперь у нас набор из (x2, y2), (x3, y3) и (x4, y4). По ним тоже проводим
    # параболу.В области пересечения парабол - на отрезке [x2, x3] - значения двух парабол осредняются.

    # Сначала определяем, между какими точками заданной сетки находится целевая точка:

    i = get_left_i(x_set, x_target)

    # Теперь мы точно знаем, что aim_x лежит между  x_set[i] и  x_set[i + 1]

    # Осталось построить параболы. Если точка находится около краёв, то парабола будет одна. Если же она где-то внутри
    # сетки, то парабол будет две, и значение в aim_x придётся усреднять. Заведём список, в который мы запишем значения
    # от парабол:

    results = []

    # И с проверками будем строить парбаолы:

    if i >= 1:

        results.append(parabola([[x_set[i - 1], z_set[i - 1]],
                                 [x_set[i], z_set[i]],
                                 [x_set[i + 1], z_set[i + 1]]], x_target))

    if i < (x_set.shape[0] - 2):

        results.append(parabola([[x_set[i], z_set[i]],
                                 [x_set[i + 1], z_set[i + 1]],
                                 [x_set[i + 2], z_set[i + 2]]], x_target))

    # Возвращаем среднее:

    return np.average(results)


def two_dim_parab_inter_surf(x_set, y_set, z_set, new_x_set, new_y_set):
    """

    :param x_set: strictly ascending 1D-numerical array
    :param y_set: strictly ascending 1D-numerical array
    :param z_set: 2D-numerical array of corresponding values z = z(x, y)
    :param new_x_set: strictly ascending 1D-numerical array corresponding to x_set with new sample rate
    :param new_y_set: strictly ascending 1D-numerical array corresponding to y_set with new sample rate
    :return: 2D-numeric array of values of z(x, y) function interpolated on the new grid
    """
    # Строит двумерную осреднённо-параболическую интерполяцию дискретно заданной функции двух переменных на новую сетку
    # координат new_x_set, new_y_set. Двумерная - т.е. сначала с помощью одномерной интерполяции строим разрез
    # имеющиейся поверхности вдоль прямой x = new_x[i], а затем по полученному разрезу строим одномерную интерполяцию
    # в точку [new_x[i], new_y[i]].

    new_f = np.zeros((new_x_set.shape[0], new_y_set.shape[0]))  # сюда будем записывать значения интерполированной
    # функции на новой сетке

    #     Итого: new_set_1 = z(aim_x, y_i), new_set_2 = z(x_i, aim_y)

    for i in range(new_x_set.shape[0]):

        crosssect = np.zeros(y_set.shape[0]) # разрез x = new_x[i]. Заполним его:

        for q in range(y_set.shape[0]):

            crosssect[q] = one_dim_parab_inter(x_set, z_set[:, q], new_x_set[i])  # построили разрез x = new_x[i]

        for j in range(new_y_set.shape[0]):

            new_f[i, j] = one_dim_parab_inter(y_set, crosssect, new_y_set[j])  # и вдоль этого разреза интерполируем в
            # y = new_y[j]

    return new_f
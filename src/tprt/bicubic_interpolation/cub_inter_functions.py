# Источник на применяемую теорию: Марчук Г.И. Методы вычислительной математики. М.: Наука. Главная редакция физико-
# математической литературы, 1980

import numpy as np

from scipy.linalg import solve_banded


def get_left_i(x_set, x):

    # Ищет номер ближайшего слева к x элемента из x_set. Т.е. x - число, а x_set - упорядоченный по возрастанию массив
    # чисел. Взято отсюда: https://www.geeksforgeeks.org/find-closest-number-array/ (с модификациями)

    # Граничные случаи:
    if x <= x_set[0]:
        return 0
    if x >= x_set[- 1]:
        return x_set.shape[0] - 2

    # Переходим к бинарному поиску
    i = 0
    j = x_set.shape[0]
    mid = 0

    while (i < j):

        mid = round((i + j) / 2) # идём на середину

        # Проверяем, не попали ли мы уже на нужный нам элемент
        if (x_set[mid] == x):

            return mid

        # Если наш элемент меньше, чем срединный, ищем в левой половине.

        if x < x_set[mid]:

            # Cначала проверяем, не является ли наш элемент граничным случаем для этой половины:

            if mid > 0 and x_set[mid - 1] < x:

                return mid - 1

            # Приписываем значение mid правой границе исследуемой области и идём на следующую итерацию:

            j = mid

        # Если же мы оказались справа от середины:

        else:

            # Аналогичная проверка на "граничность":

            if mid < x_set.shape[0] - 1 and x < x_set[mid + 1]:

                return mid

            # Аналогично, сдвигаем левую границу к mid:

            i = mid

    # В конце цикла останется единственный элемент:

    return mid

# 1D-interpolation functions:


def second_der_set(x_set, z_set):

    # Returns array of second derivatives of z in x = x_set[i].

    # Here a system of linear equations is solved:
    # A m = B
    # Matrix A is three-diagonal. Som there is no need to keep it all.

    A = np.zeros((3, z_set.shape[0] - 2)) # here we shall write all three non-zero diagonals of the system's matrix
    B = np.zeros(A.shape[1]) # right part of the system

    A[1, 0] = (x_set[2] - x_set[0]) / 3
    A[2, 0] = (x_set[2] - x_set[1]) / 6

    A[0, -1] = (x_set[-2] - x_set[-3]) / 6
    A[1, -1] = (x_set[-1] - x_set[-3]) / 3

    B[0] = (z_set[2] - z_set[1]) / (x_set[2] - x_set[1]) - (z_set[1] - z_set[0]) / (x_set[1] - x_set[0])

    B[-1] = (z_set[- 1] - z_set[- 2]) / (x_set[- 1] - x_set[- 2]) -\
            (z_set[- 2] - z_set[- 3]) / (x_set[- 2] - x_set[- 3])

    for i in np.arange(1, A.shape[1] - 1, 1):

        A[0, i] = (x_set[i + 1] - x_set[i]) / 6
        A[1, i] = (x_set[i + 2] - x_set[i]) / 3
        A[2, i] = (x_set[i + 1] - x_set[i]) / 6

        B[i] = (z_set[i + 2] - z_set[i + 1]) / (x_set[i + 2] - x_set[i + 1]) -\
               (z_set[i + 1] - z_set[i]) / (x_set[i + 1] - x_set[i])

    sec_deriv = solve_banded((1, 1), A, B) # here are second derivatives in points x[1]...x[- 2].
    # So, let's append left and right edges which are supposed to be equal to zero:

    sec_deriv = np.append(0, sec_deriv)
    sec_deriv = np.append(sec_deriv, 0)

    return sec_deriv


def one_dim_polynomial(x_set, z_set):

    # Returns array of coefficients of the interpolation ploynomials for function z_set defined on grid x_set with
    # second derivatives at the edges given.

    one_dim_coefficients = np.zeros((x_set.shape[0] - 1, 4)) # first index represents x_i (nearest left neighbour of
    # current x) and the second indicates the power of x in the polynomial.

    m_i = second_der_set(x_set, z_set)

    for i in np.arange(1, x_set.shape[0], 1):

        step_i = x_set[i] - x_set[i - 1]

        one_dim_coefficients[i - 1, 0] = m_i[i - 1] * x_set[i] ** 3 / (6 * step_i) - \
                                         m_i[i] * x_set[i - 1] ** 3 / (6 * step_i) + \
                                         (z_set[i - 1] - m_i[i - 1] * step_i ** 2 / 6) * x_set[i] / step_i - \
                                         (z_set[i] - m_i[i] * step_i ** 2 / 6) * x_set[i - 1] / step_i

        one_dim_coefficients[i - 1, 1] = - m_i[i - 1] * 3 * x_set[i] ** 2 / (6 * step_i) + \
                                         m_i[i] * 3 * x_set[i - 1] ** 2 / (6 * step_i) - \
                                         (z_set[i - 1] - m_i[i - 1] * step_i ** 2 / 6) / step_i + \
                                         (z_set[i] - m_i[i] * step_i ** 2 / 6) / step_i

        one_dim_coefficients[i - 1, 2] = m_i[i - 1] * 3 * x_set[i] / (6 * step_i) - \
                                         m_i[i] * 3 * x_set[i - 1] / (6 * step_i)

        one_dim_coefficients[i - 1, 3] = - m_i[i - 1] / (6 * step_i) + \
                                         m_i[i] / (6 * step_i)

    return one_dim_coefficients

# 2D-interpolation functions


def two_dim_polynomial(x_set, y_set, z_set):

    # Returns array of polynomial coefficients for bicubic interpolation.

    # We'll need several arrays of coefficients that can be found using one-dimensional interpolation:

    # 1. Array of polynoial coefficients for cross-sections z(xi, y):

    xi_coeff = np.zeros((x_set.shape[0], y_set.shape[0] - 1, 4))

    for i in range(x_set.shape[0]):

        xi_coeff[i, :, :] = one_dim_polynomial(y_set, z_set[i, :])

    # 2. Array of polynomial coefficients for cross-section of the second partial derivative of z with respect to x
    # along lines x = xi (so, this function will be a cubic polynomial of y):

    # First, we have to find function d2z / dx2 on the grid:

    z_xx = np.zeros((x_set.shape[0], y_set.shape[0])) # array for d2z/dx2

    for j in range(y_set.shape[0]):

        z_xx[:, j] = second_der_set(x_set, z_set[:, j])

    # And now - let's find polynomial coefficients:

    z_xx_coeff = np.zeros((x_set.shape[0], y_set.shape[0] - 1, 4))

    for i in range(x_set.shape[0]):

        z_xx_coeff[i, :, :] = one_dim_polynomial(y_set, z_xx[i, :])

    # Now we have everything what we need. Let's construct new array for two-dimensional interpolation polynomial
    # coefficients:

    two_dim_coefficients = np.zeros((x_set.shape[0] - 1, y_set.shape[0] - 1, 4, 4))

    for i in np.arange(1, x_set.shape[0], 1):

        step_i = x_set[i] - x_set[i - 1]

        for j in np.arange(1, y_set.shape[0], 1):

            for m in range(4):

                two_dim_coefficients[i - 1, j - 1, 0, m] = (xi_coeff[i - 1, j - 1, m] * x_set[i] -
                                                            xi_coeff[i, j - 1, m] * x_set[i - 1]) / step_i -\
                                                           step_i * (z_xx_coeff[i - 1, j - 1, m] * x_set[i] -
                                                                     z_xx_coeff[i, j - 1, m] * x_set[i - 1]) / 6 +\
                                                           (z_xx_coeff[i - 1, j - 1, m] * x_set[i] ** 3 -
                                                            z_xx_coeff[i, j - 1, m] * x_set[i - 1] ** 3) / step_i / 6

                two_dim_coefficients[i - 1, j - 1, 1, m] = (xi_coeff[i, j - 1, m] -
                                                            xi_coeff[i - 1, j - 1, m]) / step_i -\
                                                           step_i * (z_xx_coeff[i, j - 1, m] -
                                                                     z_xx_coeff[i - 1, j - 1, m]) / 6 +\
                                                           (z_xx_coeff[i, j - 1, m] * x_set[i - 1] ** 2 -
                                                            z_xx_coeff[i - 1, j - 1, m] * x_set[i] ** 2) / step_i / 2

                two_dim_coefficients[i - 1, j - 1, 2, m] = (z_xx_coeff[i - 1, j - 1, m] * x_set[i] -
                                                            z_xx_coeff[i, j - 1, m] * x_set[i - 1]) / step_i / 2

                two_dim_coefficients[i - 1, j - 1, 3, m] = (z_xx_coeff[i, j - 1, m] -
                                                            z_xx_coeff[i - 1, j - 1, m]) / step_i / 6

    return two_dim_coefficients


def two_dim_inter(two_dim_coeff, x_set, y_set, x, y):

    # Returns value of z(x, y) corresponding to the pre-constructed array of polynomial coefficients and the
    # parametrization grid.

    i = get_left_i(x_set, x)
    j = get_left_i(y_set, y)

    z = 0 # this is the value of z(x, y). We shall compute it in a cycle below.

    for m in np.arange(0, 4, 1):
        for n in np.arange(0, 4, 1):

            z = z + two_dim_coeff[i, j, m, n] * ( x ** m ) * ( y ** n )

    return z


def two_dim_inter_dx(two_dim_coeff, x_set, y_set, x, y):

    # Returns partial derivative dz(x, y) / dx corresponding to the pre-constructed array of polynomial
    # coefficients and the parametrization grid.

    i = get_left_i(x_set, x)
    j = get_left_i(y_set, y)

    z_x = 0 # this is the value of dz(x, y) / dx. We shall compute it in a cycle below.

    for m in np.arange(1, 4, 1):
        for n in np.arange(0, 4, 1):

            z_x = z_x + two_dim_coeff[i, j, m, n] * ( m * x ** (m - 1) ) * ( y ** n )

    return z_x


def two_dim_inter_dy(two_dim_coeff, x_set, y_set, x, y):

    # Returns partial derivative dz(x, y) / dy corresponding to the pre-constructed array of polynomial
    # coefficients and the parametrization grid.

    i = get_left_i(x_set, x)
    j = get_left_i(y_set, y)

    z_y = 0 # this is the value of dz(x, y) / dy. We shall compute it in a cycle below.

    for m in np.arange(0, 4, 1):
        for n in np.arange(1, 4, 1):

            z_y = z_y + two_dim_coeff[i, j, m, n] * ( x ** m ) * ( n * y ** (n - 1) )

    return z_y


def two_dim_inter_dx_dx(two_dim_coeff, x_set, y_set, x, y):

    # Returns partial derivative d^2z(x, y) / dx^2 corresponding to the pre-constructed array of polynomial
    # coefficients and the parametrization grid.

    i = get_left_i(x_set, x)
    j = get_left_i(y_set, y)

    z_xx = 0 # this is the value of dz(x, y) / dy. We shall compute it in a cycle below.

    for m in np.arange(2, 4, 1):
        for n in np.arange(0, 4, 1):

            z_xx = z_xx + two_dim_coeff[i, j, m, n] * ( m * (m - 1) * x ** (m - 2) ) * ( y ** n )

    return z_xx

def two_dim_inter_dy_dy(two_dim_coeff, x_set, y_set, x, y):

    # Returns partial derivative d^2z(x, y) / dy^2 corresponding to the pre-constructed array of polynomial
    # coefficients and the parametrization grid.

    i = get_left_i(x_set, x)
    j = get_left_i(y_set, y)

    z_yy = 0 # this is the value of dz(x, y) / dy. We shall compute it in a cycle below.

    for m in np.arange(0, 4, 1):
        for n in np.arange(2, 4, 1):

            z_yy = z_yy + two_dim_coeff[i, j, m, n] * ( x ** m ) * ( n * (n - 1) * y ** (n - 2) )

    return z_yy


def two_dim_inter_dx_dy(two_dim_coeff, x_set, y_set, x, y):

    # Returns partial derivative d^2z(x, y) / dx dy corresponding to the pre-constructed array of polynomial
    # coefficients and the parametrization grid.

    i = get_left_i(x_set, x)
    j = get_left_i(y_set, y)

    z_y = 0 # this is the value of dz(x, y) / dy. We shall compute it in a cycle below.

    for m in np.arange(1, 4, 1):
        for n in np.arange(1, 4, 1):

            z_y = z_y + two_dim_coeff[i, j, m, n] * ( m * x ** (m - 1) ) * ( n * y ** (n - 1) )

    return z_y

# Потребуется минимизация, и в силу особенностей синтаксиса надо определить ещё одну функцию. За подробностями - в
# классе "GridHorizon".


def difference(s, x_set, y_set, two_dim_coeff, sou, vec):

    return abs(two_dim_inter(two_dim_coeff, x_set, y_set, sou[0] + s * vec[0], sou[1] + s * vec[1]) - \
               (sou[2] + s * vec[2]))


def parabola(points, aim_x):
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

    return A * aim_x**2 + B * aim_x + C


def one_dim_parab_inter(x_set, func, aim_x):
    # По заданной сетке и дискретно заданной функции func = np.array([f1, f2, ..., fn]) строит одномерную
    # усреднённо-параболическую интерполяцию в точке с координатой aim_x. Усреднённо-параболическая интерполяция
    # состоит в следующем: пусть есть три точки (x1, y1), (x2, y2) и (x3, y3). Через них проводится парабола. Далее
    # смешаемся на одну точку вправо: теперь у нас набор из (x2, y2), (x3, y3) и (x4, y4). По ним тоже проводим
    # параболу.В области пересечения парабол - на отрезке [x2, x3] - значения двух парабол осредняются.

    # Сначала определяем, между какими точками заданной сетки находится целевая точка:

    i = get_left_i(x_set, aim_x)

    # Теперь мы точно знаем, что aim_x лежит между  x_set[i] и  x_set[i + 1]

    # Осталось построить параболы. Если точка находится около краёв, то парабола будет одна. Если же она где-то внутри
    # сетки, то парабол будет две, и значение в aim_x придётся усреднять. Заведём список, в который мы запишем значения
    # от парабол:

    results = []

    # И с проверками будем строить парбаолы:

    if i >= 1:

        results.append(parabola([[x_set[i - 1], func[i - 1]],
                                 [x_set[i], func[i]],
                                 [x_set[i + 1], func[i + 1]]], aim_x))

    if i < (x_set.shape[0] - 2):

        results.append(parabola([[x_set[i], func[i]],
                                 [x_set[i + 1], func[i + 1]],
                                 [x_set[i + 2], func[i + 2]]], aim_x))

    # Возвращаем среднее:

    return np.average(results)


def two_dim_parab_inter_surf(x_set, y_set, f, new_x_set, new_y_set):
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

            crosssect[q] = one_dim_parab_inter(x_set, f[:, q], new_x_set[i])  # построили разрез x = new_x[i]

        for j in range(new_y_set.shape[0]):

            new_f[i, j] = one_dim_parab_inter(y_set, crosssect, new_y_set[j])  # и вдоль этого разреза интерполируем в
            # y = new_y[j]

    return new_f
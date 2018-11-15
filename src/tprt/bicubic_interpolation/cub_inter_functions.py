# Источник на применяемую теорию: http://statistica.ru/branches-maths/interpolyatsiya-splaynami-teor-osnovy/

import numpy as np


def derivatives(func, h):
    # по дискретно заданной функции (не менее четырёх точек) func = np.array([f1, f2, ..., fn]) и шагу
    # дискретизации h строит вектор производных в этих точках (методом конечных разностей)
    
    der = np.zeros(func.shape[0])
    
    der[0] = 1/(6 * h) * (- 11*func[0] + 18*func[1] - 9*func[2] + 2*func[3])
    der[-1] = 1/(6 * h) * (11*func[-1] - 18*func[-2] + 9*func[-3] - 2*func[-4])
    
    A = np.zeros((func.shape[0] - 2, func.shape[0] - 2)) # матрица системы для нахождения der_i
    B = np.zeros(func.shape[0] - 2) # правая часть этой системы
    
    A[0, 0] = 4
    A[0, 1] = 1

    B[0] = 3 * (func[2] - func[0]) / h - der[0]

    A[-1, -1] = 4
    A[-1, -2] = 1

    B[-1] = 3 * (func[-1] - func[-3]) / h - der[-1]
    
    # заполняем систему
    for i in np.arange(1, func.shape[0] - 3,1):
    
        A[i, i - 1] = 1
        A[i, i] = 4
        A[i, i + 1] = 1
        
        B[i] = 3 * (func[i + 2] - func[i]) / h
        
    # Решаем систему и сразу заносим найденные значения der_i в соотв. вектор

    der[1: func.shape[0] -1] = np.linalg.solve(A, B)
    
    return der


def one_dim_inter(x_net, fun, deriv, x_desir): # считает значение интерполированной функции fun,
    # заданной дискретно на сетке x_net, в точке x_desir. Также необходимо заранее создать и подать на вход
    # массив производных deriv.

    # надо определить, между какими значениями сетки находится точка x_desir:

    i = 0

    for k in range(x_net.shape[0] - 1):
        if x_net[k] <= x_desir <= x_net[k + 1]:
            i = k
            break

    x_i = x_net[i]
    x_i_1 = x_net[i + 1]

    # и найти шаг по сетке:

    step = x_net[1] - x_net[0]

    # теперь зададим члены полинома в интересующей нас точке:

    first_term = ((x_i_1 - x_desir)**2)*(2 * (x_desir - x_i) + step) / step**3
    second_term = ((x_desir - x_i)**2)*(2 * (x_i_1 - x_desir) + step) / step**3

    third_term = ((x_i_1 - x_desir)**2)*(x_desir - x_i) / step**2
    fourth_term = ((x_desir - x_i)**2)*(x_desir - x_i_1) / step**2

    # и вернём его значение:

    return first_term * fun[i] + second_term * fun[i + 1] + \
           third_term * deriv[i] + fourth_term * deriv[i + 1]


def one_dim_inter_ddx(x_net, fun, deriv, x_desir):
    # Считает производную функции z(x) - интерполированной fun - в точке x_desir. Опять же, на вход подаётся массив
    # производных deriv.

    step = x_net[1] - x_net[0] # шаг по сетке

    # надо определить, между какими значениями сетки находится точка x_desir:

    i = 0

    for k in range(x_net.shape[0] - 1):
        if x_net[k] <= x_desir <= x_net[k + 1]:
            i = k
            break

    x_i = x_net[i]
    x_i_1 = x_net[i + 1]

    # теперь зададим члены полинома в интересующей нас точке:

    first_term_1 = - 2 * (x_i_1 - x_desir) * (2 * (x_desir - x_i) + step) / step**3
    first_term_2 = ((x_i_1 - x_desir)**2) * 2 / step**3

    second_term_1 = 2 * (x_desir - x_i) * (2 * (x_i_1 - x_desir) + step) / step**3
    second_term_2 = ((x_desir - x_i)**2) * (- 2) / step**3

    third_term_1 = - 2 * (x_i_1 - x_desir) * (x_desir - x_i) / step**2
    third_term_2 = ((x_i_1 - x_desir)**2) / step**2

    fourth_term_1 = 2 * (x_desir - x_i) * (x_desir - x_i_1) / step**2
    fourth_term_2 = ((x_desir - x_i)**2) / step**2

    # и вернём его значение:

    return (first_term_1 + first_term_2)*fun[i] + (second_term_1 + second_term_2)*fun[i + 1] + \
           (third_term_1 + third_term_2)*deriv[i] + (fourth_term_1 + fourth_term_2)*deriv[i + 1]


def one_dim_inter_ddx2(x_net, fun, deriv, x_desir):
    # Считает вторую производную функции z(x) - интерполированной fun - в точке x_desir. deriv - массив производных.

    step = x_net[1] - x_net[0] # шаг по сетке

    # надо определить, между какими значениями сетки находится точка x_desir:

    i = 0

    for k in range(x_net.shape[0] - 1):
        if x_net[k] <= x_desir <= x_net[k + 1]:
            i = k
            break

    x_i = x_net[i]
    x_i_1 = x_net[i + 1]

    # теперь зададим члены полинома в интересующей нас точке:

    first_term_1_1 = 2 * (2 * (x_desir - x_i) + step) / step**3
    first_term_1_2 = - 2 * (x_i_1 - x_desir) * 2 / step**3

    first_term_2_1 = - 2 * (x_i_1 - x_desir) * 2 / step**3


    second_term_1_1 = 2 * (2 * (x_i_1 - x_desir) + step) / step**3
    second_term_1_2 = - 2 * 2 * (x_desir - x_i) / step**3

    second_term_2_1 = - 2 * 2 * (x_desir - x_i) / step**3


    third_term_1_1 = 2 * (x_desir - x_i) / step**2
    third_term_1_2 = - 2 * (x_i_1 - x_desir) / step**2

    third_term_2_1 = - 2 * (x_i_1 - x_desir) / step**2


    fourth_term_1_1 = 2 * (x_desir - x_i_1) / step**2
    fourth_term_1_2 = 2 * (x_desir - x_i) / step**2

    fourth_term_2_1 = 2 * (x_desir - x_i) / step**2

    # и вернём его значение:

    return (first_term_1_1 + first_term_1_2 + first_term_2_1) * fun[i] + \
           (second_term_1_1 + second_term_1_2 + second_term_2_1) * fun[i + 1] + \
           (third_term_1_1 + third_term_1_2 + third_term_2_1) * deriv[i] + \
           (fourth_term_1_1 + fourth_term_1_2 + fourth_term_2_1) * deriv[i + 1]


def two_dim_inter(x_set, y_set, f, deriv_x, x_search, y_search):
    #     на вход принимает исходные сетки по X и по Y, дискретно заданную функцию и координаты точки,
    # для которой надо будет рассчитать интерполированную функцию. Массив deriv_x - двумерный массив,
    # в котором лежат частные производные функции f по соответсвтующей переменной в каждой точке [x_set[i], y_set[j]].

    # надо посчитать значения функции f на прямой x_search

    new_f = np.zeros(y_set.shape[0])

    for q in range(y_set.shape[0]):

        new_f[q] = one_dim_inter(x_set, f[:, q], deriv_x[:, q], x_search)

    x_step = x_set[1] - x_set[0] # шаг по X

    deriv_y = derivatives(new_f, x_step) # частные производнц по Y на нужном разрезе

    f_search = one_dim_inter(y_set, new_f, deriv_y, y_search)

    return f_search


def two_dim_inter_surf(x_set, y_set, f, deriv_x, new_x_set, new_y_set):
    # На вход принимает исходные сетки по X и по Y, дискретно заданную функцию и массив deriv_x - двумерный массив,
    # в котором лежат частные производные функции f по соответсвтующей переменной в каждой точке [x_set[i], y_set[j]].
    # Последние два аргумента задают новую сетку, в уздах которой надо будет посчитать значения интерполированной функции.
    # Возвращает массив этих интерполированных значений.

    y_step = y_set[1] - y_set[0] # шаг исходной сетки по Y понадобится потом

    new_f = np.zeros((new_x_set.shape[0], new_y_set.shape[0])) # сюда будем записывать значения интерполированной функции
    # на новой сетке

    for i in range(new_x_set.shape[0]):

        crosssect_x = np.zeros(y_set.shape[0]) # разрез x = new_x[i]. Заполним его:

        for q in range(y_set.shape[0]):

            crosssect_x[q] = one_dim_inter(x_set, f[:, q], deriv_x[:, q], new_x_set[i]) # построили разрез x = new_x[i]


        deriv_y = derivatives(crosssect_x, y_step) # частные производне по Y на нужном разрезе


        for j in range(new_y_set.shape[0]):

            new_f[i, j] = one_dim_inter(y_set, crosssect_x, deriv_y, new_y_set[j]) # и вдоль этого разреза интерполируем в
            # y = new_y[j]

    return new_f


# потребуется минимизация, и в силу особенностей синтаксиса надо определить ещё одну функцию. За подробностями - в
# классе "GridHorizon".
def difference(s, x_set, y_set, func, deriv_x, sou, vec):

    return abs(two_dim_inter(x_set, y_set, func, deriv_x, sou[0] + s * vec[0], sou[1] + s * vec[1]) - \
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

    i = 0

    for k in range(x_set.shape[0] - 1):
        if aim_x >= x_set[k] and aim_x <= x_set[k + 1]:
            i = k
            break

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

    #         Возвращаем не только среднее, но и значения от обеих парабол по отдельности:

    return np.average(results), results


def two_dim_parab_inter_surf(x_set, y_set, f, new_x_set, new_y_set):
    # Строит двумерную осреднённо-параболическую интерполяцию дискретно заданной функции двух переменных на новую сетку
    # координат new_x_set, new_y_set. Двумерная - т.е. сначала с помощью одномерной интерполяции строим разрез
    # имеющиейся поверхности вдоль прямой x = new_x[i], а затем по полученному разрезу строим одномерную интерполяцию
    # в точку [new_x[i], new_y[i]].

    new_f = np.zeros((new_x_set.shape[0], new_y_set.shape[0])) # сюда будем записывать значения интерполированной
    # функции на новой сетке

    #     Итого: new_set_1 = z(aim_x, y_i), new_set_2 = z(x_i, aim_y)

    for i in range(new_x_set.shape[0]):

        crosssect = np.zeros(y_set.shape[0]) # разрез x = new_x[i]. Заполним его:

        for q in range(y_set.shape[0]):

            crosssect[q] = one_dim_parab_inter(x_set, f[:, q], new_x_set[i])[0] # построили разрез x = new_x[i]


        for j in range(new_y_set.shape[0]):

            new_f[i, j] = one_dim_parab_inter(y_set, crosssect, new_y_set[j])[0] # и вдоль этого разреза интерполируем в
            # y = new_y[j]

    return new_f
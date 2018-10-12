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


def one_dim_inter(x_set, func, aim_x):
    # считает значение интерполяции функции func = np.array([f1, f2, ..., fn]), заданной дискретно на сетке
    # x_set = np.array([x1, x2, ...,xn]), в точке aim_x, находящейся В ПРЕДЕЛАХ сетки. Проверки на выолнение этого
    # условия нет.
    
    step = x_set[1] - x_set[0] # шаг по сетке
    deriv = derivatives(func, step) # вектор производных
    
    # надо определить, между какими значениями сетки находится точка aim_x:
    
    i = 0
    
    for k in range(x_set.shape[0] - 1):
        if aim_x >= x_set[k] and aim_x <= x_set[k + 1]:
            i = k
            break
            
    x_i = x_set[i]
    x_i_1 = x_set[i + 1] # будет удобнее записать значения x_i и x_i+1, между которыми находится aim_x
    
    # теперь зададим члены полинома в интересующей нас точке:
    
    first_term = ((x_i_1 - aim_x)**2)*(2 * (aim_x - x_i) + step) / step**3
    second_term = ((aim_x - x_i)**2)*(2 * (x_i_1 - aim_x) + step) / step**3
    
    third_term = ((x_i_1 - aim_x)**2)*(aim_x - x_i) / step**2
    fourth_term = ((aim_x - x_i)**2)*(aim_x - x_i_1) / step**2
    
    # и вернём его значение:
    
    return first_term * func[i] + second_term * func[i + 1] + third_term * deriv[i] + fourth_term * deriv[i + 1]

def one_dim_inter_ddx(x_net, fun, x_desir):
    #     Считает производную функции z(x) - интерполированной fun - в точке x_desir

    step = x_net[1] - x_net[0] # шаг по сетке
    deriv = derivatives(fun, step) # вектор производных

    # надо определить, между какими значениями сетки находится точка x_desir:

    i = 0

    for k in range(x_net.shape[0] - 1):
        if x_desir >= x_net[k] and x_desir <= x_net[k + 1]:
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


def two_dim_inter(x_set, y_set, func, aim_x, aim_y):
    # на вход принимает исходные сетки по X и по Y, дискретно заданную функцию func = func(X, Y) = {finc[i, j]} и
    # координаты точки, для которой надо будет рассчитать значение интерполяции функции Z
                    
    # надо посчитать значения функции func на прямой x = aim_x
            
    new_func = np.zeros(y_set.shape[0])
                
    for q in range(y_set.shape[0]):
                
        new_func[q] = one_dim_inter(x_set, func[:, q], aim_x) # буквально: проинтерполировали все строчки в функции
        # func[i, j], получили дискретно заданную функцию одной переменной: new_func = new_func(aim_x, y_set)
            
    aim_func = one_dim_inter(y_set, new_func, aim_y) # и эту новую функцию проинтерполировали в точку aim_y
            
    return aim_func


# потребуется минимизация, и в силу особенностей синтаксиса надо определить ещё одну функцию
def difference(x, sou,x_set, y_set, func, alfa, beta, gamma, omega):

    return abs(two_dim_inter(x_set, y_set, func, x, alfa*x + beta) - \
               np.sign(x - sou[0]) * \
               gamma*np.sqrt((alfa**2 + 1)*x**2 + \
                             (2*alfa*(beta - sou[1]) - 2*sou[0]) * x + \
                             (beta - sou[1])**2 + sou[0]**2) - omega)
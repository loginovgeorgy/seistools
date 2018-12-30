# Локальная система координат: ось X направлена вправо, ось Z - вниз, а ось Y - в плоскость экрана на нас.

# Названия "S1" и "S2" означаеют соответственно "SV" и "SH" (комментарии в большинсвте своём писались в ходе работы,
# и поэтому было легче пользоваться индексацией, применяемой в коде).

# Аргументы подаются в следующем порядке:

# layer1 - слой, в котором распространяется падающая волна
# layer2 - подстилающий его слой

# cos_inc - косинус угла падения
# inc_polariz - полный вектор поляризации падающей волны, записанный уже в локольных координатах.
# inc_vel - скорость распространеня падающей волны

# rt_signum - указывает на то, что происходит на границе - отражение, или преломление:
# + 1 - преломление,
# - 1 - отражение

# На выходе функция выдаёт коэффициент отражения / прохождения для указанной волны.

import cmath as cm

from .polarizations import *


def rt_coefficients(layer1, layer2, cos_inc, inc_polariz, inc_vel, rt_signum):

    # print("\x1b[1;30m velp1 = ", layer1.get_velocity(0)['vp'])
    # print("\x1b[1;30m velp2 = ", layer2.get_velocity(0)['vp'])
    # print("\x1b[1;30m cos_inc = ", cos_inc)
    # print("\x1b[1;30m inc_polariz = ", inc_polariz)
    # print("\x1b[1;30m inc_vel = ", inc_vel)
    # print("\x1b[1;30m rt_signum = ", rt_signum, '\n')

    # Для решения поставленной задачи потребуются матрицы упругих модулей сред 1 и 2:
    c_ij1 = c_ij([layer1.get_velocity(0)['vp'], layer1.get_velocity(0)['vs']], layer1.density / 1000)
    c_ij2 = c_ij([layer2.get_velocity(0)['vp'], layer2.get_velocity(0)['vs']], layer2.density / 1000)
    # Делим на 1000, чтобы избежать зашкаливающе больших чисел и соответствующих ошибок. На решения системы такая
    # нормировка не повлияет.

    # сформируем данные о падающей волне, её волновой вектор k0, её поляризацию и вектор медленности:
    k0 = np.array([np.sqrt(1 - cos_inc ** 2), 0, cos_inc])  # нормаль к фронту

    v0 = inc_vel
    u0 = inc_polariz / np.linalg.norm(inc_polariz) # нормируем, т.к. все остальные векторы будут единичными
    p0 = k0 / v0  # и её вектор медленности

    # Создадим два массива. В первом будут лежать волновые векторы-строки отражённых волн, а во втором -
    # преломлённых.

    k_refl = np.array([np.zeros(3, dtype = complex), # P-волна
                       np.zeros(3, dtype = complex), # SV-волна
                       np.zeros(3, dtype = complex)]) # SH-волна

    k_trans = np.array([np.zeros(3, dtype = complex), # P-волна
                        np.zeros(3, dtype = complex), # SV-волна
                        np.zeros(3, dtype = complex)]) # SH-волна

    # Зададим сначала все волновые векторы в соответствии с законом Снеллиуса:

    k_refl[0] = np.array([k0[0] * layer1.get_velocity(0)['vp'] / v0,
                          0,
                          - cm.sqrt(1 - (k0[0] * layer1.get_velocity(0)['vp'] / v0) ** 2)])

    k_refl[1] = np.array([k0[0] * layer1.get_velocity(0)['vs'] / v0,
                          0,
                          - cm.sqrt(1 - (k0[0] * layer1.get_velocity(0)['vs'] / v0) ** 2)])
    k_refl[2] = k_refl[1] # волновые векторы для SV- и SH-волн в изотропной среде совпадают

    k_trans[0] = np.array([k0[0] * layer2.get_velocity(0)['vp'] / v0,
                           0,
                           cm.sqrt(1 - (k0[0] * layer2.get_velocity(0)['vp'] / v0) ** 2)])

    k_trans[1] = np.array([k0[0] * layer2.get_velocity(0)['vs'] / v0,
                           0,
                           cm.sqrt(1 - (k0[0] * layer2.get_velocity(0)['vs'] / v0) ** 2)])
    k_trans[2] = k_trans[1] # волновые векторы для SV- и SH-волн в изотропной среде совпадают

    # Теперь заводим векторы медленности:

    # отражённые волны
    p_refl_p = k_refl[0] / layer1.get_velocity(0)['vp']

    p_refl_s1 = k_refl[1] / layer1.get_velocity(0)['vs']

    p_refl_s2 = k_refl[2] / layer1.get_velocity(0)['vs']

    # преломлённые волны

    p_trans_p = k_trans[0] / layer2.get_velocity(0)['vp']

    p_trans_s1 = k_trans[1] / layer2.get_velocity(0)['vs']

    p_trans_s2 = k_trans[2] / layer2.get_velocity(0)['vs']
        
    # Поляризации отражённых и преломлённых волн:
    # отражённые волны
    
    u = polarizations(c_ij1, p_refl_p)  # находим, с какими поляризациями волна может распространяться
    # в 1-й среде при заданном веторе медленности p_refl_p
    
    u_refl_p = u[:, 0]  # по построению, поляризация продольной волны - "первая в списке".
    
    u = polarizations(c_ij1, p_refl_s1)
    
    u_refl_s1 = u[:, 1]
    u_refl_s2 = u[:, 2]  # считать отдельно матрицы v и u для волны S2 бессмысленно, т.к. нормаль к её фронту
    # совпадает с нормалью к фронту волны S1

    # преломлённые волны
    
    u = polarizations(c_ij2, p_trans_p)  # находим, с какими поляризациями волна может распространяться
    # в 1-й среде при заданном веторе медленности p_refl_p
    
    u_trans_p = u[:, 0]  # по построению, поляризация продольной волны - "первая в списке".
    
    u = polarizations(c_ij2, p_trans_s1)
    
    u_trans_s1 = u[:, 1]
    u_trans_s2 = u[:, 2]  # считать отдельно матрицы v и u для волны S2 бессмысленно, т.к. нормаль к её фронту
    # совпадает с нормалью к фронту волны S1

    # Зададим матрицу системы уравнений на границе.
    
    # Первые три уравнения задаются легко. А вот оставшиеся три мы будем "вбивать" не напрямую, а по алгоритму,
    # представленому в "Лучевой метод в анизотропной среде (алгоритмы, программы)" Оболенцева, Гречки на стр. 97.
    # Т.е. будем задавать коэффициенты системы через довольно хитрые циклы.
    # Кроме всего прочего, этот алгоритм, вроде бы, универсален и для изотропных, и для анизотропных сред.

    # Коэффициенты в матрице могут быть и комплексными, что надо указать при задании массивов.

    # Падающая волна:
    u0_coeff = np.zeros(3, dtype = complex) #коэффициенты для падающей волны

    # Отражённые волны:
    rp_coeff = np.zeros(3, dtype = complex) #коэффициенты для отражённой P-волны
    rs1_coeff = np.zeros(3, dtype = complex) #коэффициенты для отражённой S1-волны
    rs2_coeff = np.zeros(3, dtype = complex) #коэффициенты для отражённой S2-волны

    # Преломлённые волны:
    tp_coeff = np.zeros(3, dtype = complex) #коэффициенты для преломлённой P-волны
    ts1_coeff = np.zeros(3, dtype = complex) #коэффициенты для преломлённой S1-волны
    ts2_coeff = np.zeros(3, dtype = complex) #коэффициенты для преломлённой S2-волны

    # При задании системы понадобится полный тензор упругих модулей:

    c_ijkl_1 = c_ijkl(c_ij1)
    c_ijkl_2 = c_ijkl(c_ij2)

    # Заполненяем в цикле векторы коэффициентов системы:

    for i in range(3):
        for j in range(3):
            for k in range(3):

                u0_coeff[i] = u0_coeff[i] +\
                              c_ijkl_1[i, 2, j, k] * p0[j] * u0[k]

                rp_coeff[i] = rp_coeff[i] +\
                              c_ijkl_1[i, 2, j, k] * p_refl_p[j] * u_refl_p[k]

                rs1_coeff[i] = rs1_coeff[i] +\
                               c_ijkl_1[i, 2, j, k] * p_refl_s1[j] * u_refl_s1[k]

                rs2_coeff[i] = rs2_coeff[i] +\
                               c_ijkl_1[i, 2, j, k] * p_refl_s2[j] * u_refl_s2[k]



                tp_coeff[i] = tp_coeff[i] +\
                              c_ijkl_2[i, 2, j, k] * p_trans_p[j] * u_trans_p[k]

                ts1_coeff[i] = ts1_coeff[i] +\
                               c_ijkl_2[i, 2, j, k] * p_trans_s1[j] * u_trans_s1[k]

                ts2_coeff[i] = ts2_coeff[i] +\
                               c_ijkl_2[i, 2, j, k] * p_trans_s2[j] * u_trans_s2[k]

    # Итого:

    matrix = np.array([[u_refl_p[0], u_refl_s1[0], u_refl_s2[0],
                        - u_trans_p[0], - u_trans_s1[0], - u_trans_s2[0]],
                       [u_refl_p[1], u_refl_s1[1], u_refl_s2[1],
                        - u_trans_p[1], - u_trans_s1[1], - u_trans_s2[1]],
                       [u_refl_p[2], u_refl_s1[2], u_refl_s2[2],
                        - u_trans_p[2], - u_trans_s1[2], - u_trans_s2[2]],
                       [rp_coeff[0], rs1_coeff[0], rs2_coeff[0],
                        - tp_coeff[0], - ts1_coeff[0], - ts2_coeff[0]],
                       [rp_coeff[1], rs1_coeff[1], rs2_coeff[1],
                        - tp_coeff[1], - ts1_coeff[1], - ts2_coeff[1]],
                       [rp_coeff[2], rs1_coeff[2], rs2_coeff[2],
                        - tp_coeff[2], - ts1_coeff[2], - ts2_coeff[2]]])

    # Введём правую часть системы уравнений:

    right_part = np.array([- u0[0],
                           - u0[1],
                           - u0[2],
                           - u0_coeff[0],
                           - u0_coeff[1],
                           - u0_coeff[2]])

    # Решим эту систему уравнений:
    rp, rs1, rs2, tp, ts1, ts2 = np.linalg.solve(matrix, right_part)

    # print("\x1b[1;30m rp = ", rp)
    # print("\x1b[1;30m rs1 = ", rs1)
    # print("\x1b[1;30m rs2= ", rs2)
    # print("\x1b[1;30m tp = ", tp)
    # print("\x1b[1;30m ts1 = ", ts1)
    # print("\x1b[1;30m ts2 = ", ts2, '\n')
    # print("\x1b[1;30m matrix = ", matrix)
    # print("\x1b[1;30m right part = ", right_part, '\n')

    # И вернём результат:

    if rt_signum == 1:

        return tp, ts1, ts2

    else:

        return rp, rs1, rs2

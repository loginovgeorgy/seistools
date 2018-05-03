# В этом скрипте будет реализвана функция, честно считающая коэффициенты отражения/прохождения на границе раздела двух
# упругих сред. Под "честным" здесь понимается составление системы уравнений на основе граничных условий и её решение.

# Система координат: ось X направлена вправо, ось Z - вниз, а ось Y - в плоскость экрана на нас. Модель "классическая":
# волна падает на горизонтальную границу двух ИЗОТРОПНЫХ сред в плоскости XZ под некоторым углом падения, отсчитываемым
# против часовой стрелки от нормали "вверх" от границы раздела. Поляризации волн могут быть трёх видов: продольная
# (сонаправлена с нормалью к фронту), поперечная-1 (отклонена на 90 градусов по часовой стрелке от нормали к фронту,
# лежит в плоскости экрана) и поперечная-2 (направлена в плоскость экрана на нас).
# Таким образом, тройка поляризаций "поперечная-2, поперечная-1 и продольная" - правая.

# Названия "S1" и "S2" означаеют соответственно "SV" и "SH" (комментарии в большинсвте своём писались в ходе работы,
# и поэтому было легче пользоваться индексацией, применяемой в коде).

# На вход функции будут подаваться данные о средах над и под границей раздела, как то: плотности, скорости продольных и
# поперечных волн. Кроме того, в аргументах функции необходимо указать тип падающей волны и угол её падения.
# Аргументы подаются в следующем порядке:

# σ1 - плотность пород в первой среде (над границей), кг/м^3

# σ2 - плотность пород во второй среде (под границей), кг/м^3

# Vp1 - скорость продольных волн в первой среде, м/с

# Vs1 - скорость поперечных волн в первой среде, м/с

# Vp2 - скорость продольных волн во второй среде, м/с

# Vs2 - скорость поперечных волн во второй среде, м/с

# Wave_Type - тип падающей волны, 0 - продольная, 1 - поперечная-1, 2 - поперечная-2

# Angle_Deg - угол падения В ГРАДУСАХ

# На выходе функция выдаёт коэффициенты отражения и прохождения в следующем порядке:

# Rp - коэффициент отражения P-волны

# Rs1 - коэффициент отражения S1-волны

# Rs2 - коэффициент отражения S2-волны

# Tp - коэффициент прохождения P-волны

# Ts1 - коэффициент прохождения S1-волны

# Ts2 - коэффициент прохождения S2-волны

# ЕСТЬ СЛОЖНОСТЬ: ГДЕ-ТО ИСПОЛЬЗУЕТСЯ numpy (только действительные числа), А ГДЕ-ТО cmath (комплекснозначные функции).
# ВОЗМОЖНЫ ПРОБЛЕМЫ.

import numpy as np

from .polarizations import *


def rt_coefficients(σ1, σ2, vp1, vs1, vp2, vs2, wave__type, angle__deg):

    # для начала, нужно немного поколдовать над входными данными - перевести угол из градусов в радианы:
    angle__rad = np.radians(angle__deg)

    # далее, для решения поставленной задачи потребуются матрицы упругих модулей сред 1 и 2:
    c_ij1 = c_ij([vp1, vs1], σ1)
    c_ij2 = c_ij([vp2, vs2], σ2)

    # сформируем данные о падающей волне, найдём нормаль к её фронту, её поляризацию и вектор медленности:
    n = np.array([np.sin(angle__rad), 0, np.cos(angle__rad)])  # нормаль к фронту

    if wave__type == 0:
        v = vp1
    else:
        v = vs1
        
    u = polarizations(c_ij1, n)
    # мы задали скорости волн в 1-й среде и их поляризации.
    # Обозначения v, u будут часто использоваться в данном скрипте для разных величин,
    # так что не следует запоминать за ними конкретный физический смысл

    v0 = v  # запомним скорость падающей волны
    u0 = u[:, wave__type]
    p0 = n/v0  # и её вектор медленности
    
    
#     найдём параметры отражённых и преломлённых волн (соответственно, Refl и Trans):
   
# вектора медленности:

    # отражённые волны
    # первые две координаты сохраняются по закону Снеллиуса
    p_refl_p = np.array([p0[0], p0[1], -cm.sqrt(1/vp1**2 - p0[0]**2 - p0[1]**2)])
    # первые две координаты сохраняются по закону Снеллиуса
    p_refl_s1 = np.array([p0[0], p0[1], -cm.sqrt(1/vs1**2 - p0[0]**2 - p0[1]**2)])
    p_refl_s2 = p_refl_s1  # векторы медленности S1 и S2 волн в изотропной среде совпадают

    # преломлённые волны
    # первые две координаты сохраняются по закону Снеллиуса
    p_trans_p = np.array([p0[0], p0[1], cm.sqrt(1/vp2**2 - p0[0]**2 - p0[1]**2)])
    # первые две координаты сохраняются по закону Снеллиуса
    p_trans_s1 = np.array([p0[0], p0[1], cm.sqrt(1/vs2**2 - p0[0]**2 - p0[1]**2)])
    p_trans_s2 = p_trans_s1  # векторы медленности S1 и S2 волн в изотропной среде совпадают
        
# поляризации отражённых и преломлённых волн:
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
    
#     переходим к составлению системы уравнений

# Нужно определить символы, относительно которых будем решать уравнния
# ЗАКОММЕНТИРОВАНО, Т.К. ИСПОЛЬЗОВАНИЕ БИБЛИОТЕКИ SYMPY ЗДЕСЬ ИЗЛИШНЕ!!!

#     Rp,Rs1,Rs2,Tp,Ts1,Ts2 = sp.symbols(' Rp,Rs1,Rs2,Tp,Ts1,Ts2 ')
    
    # нам понадобятся коэффициенты Ламе

#     λ1 = c_ij1[0,0] - 2*c_ij1[3,3]
#     μ1 = c_ij1[3,3]

#     λ2 = c_ij2[0,0] - 2*c_ij2[3,3]
#     μ2 = c_ij2[3,3]
    
# #     условие №1: нулевой скачок смещений на границе:
#     #приравниваем x-координаты смещений сверху и снизу от границы:
#     Eq0 = sp.Eq(u0[0] + u_refl_p[0]*Rp + u_refl_s1[0]*Rs1 + u_refl_s2[0]*Rs2,u_trans_p[0]*Tp + u_trans_s1[0]*Ts1 + u_trans_s2[0]*Ts2)
#     #приравниваем y-координаты смещений сверху и снизу от границы:
#     Eq1 = sp.Eq(u0[1] + u_refl_p[1]*Rp + u_refl_s1[1]*Rs1 + u_refl_s2[1]*Rs2,u_trans_p[1]*Tp + u_trans_s1[1]*Ts1 + u_trans_s2[1]*Ts2)
#     #приравниваем z-координаты смещений сверху и снизу от границы:
#     Eq2 = sp.Eq(u0[2] + u_refl_p[2]*Rp + u_refl_s1[2]*Rs1 + u_refl_s2[2]*Rs2,u_trans_p[2]*Tp + u_trans_s1[2]*Ts1 + u_trans_s2[2]*Ts2)
    
#     # условие №2: нулевой скачок нормальных напряжений на границе
# # с помощью закона Гука и приближения малых деформаций это условие сводится к условиям на производные вектора смещений;
# # далее используется предположение о виде сигнала
#     # (σ13)I = (σ13)II
#     Eq3 = sp.Eq(μ1*(u0[0]*n[2]/v0 + u_refl_p[0]*nReflP[2]/vp1*Rp + u_refl_s1[0]*nReflS1[2]/vs1*Rs1 + u_refl_s2[0]*nReflS2[2]/vs1*Rs2 +\
#                     u0[2]*n[0]/v0 + u_refl_p[2]*nReflP[0]/vp1*Rp + u_refl_s1[2]*nReflS1[0]/vs1*Rs1 + u_refl_s2[2]*nReflS2[0]/vs1*Rs2),\
#                 μ2*(u_trans_p[0]*nTransP[2]/vp2*Tp + u_trans_s1[0]*nTransS1[2]/vs2*Ts1 + u_trans_s2[0]*nTransS2[2]/vs2*Ts2 +\
#                     u_trans_p[2]*nTransP[0]/vp2*Tp + u_trans_s1[2]*nTransS1[0]/vs2*Ts1 + u_trans_s2[2]*nTransS2[0]/vs2*Ts2))
#     # (σ23)I = (σ23)II
#     Eq4 = sp.Eq(μ1*(u0[1]*n[2]/v0 + u_refl_p[1]*nReflP[2]/vp1*Rp + u_refl_s1[1]*nReflS1[2]/vs1*Rs1 + u_refl_s2[1]*nReflS2[2]/vs1*Rs2 +\
#                     u0[2]*n[1]/v0 + u_refl_p[2]*nReflP[1]/vp1*Rp + u_refl_s1[2]*nReflS1[1]/vs1*Rs1 + u_refl_s2[2]*nReflS2[1]/vs1*Rs2),\
#                 μ2*(u_trans_p[1]*nTransP[2]/vp2*Tp + u_trans_s1[1]*nTransS1[2]/vs2*Ts1 + u_trans_s2[1]*nTransS2[2]/vs2*Ts2 +\
#                     u_trans_p[2]*nTransP[1]/vp2*Tp + u_trans_s1[2]*nTransS1[1]/vs2*Ts1 + u_trans_s2[2]*nTransS2[1]/vs2*Ts2))
#     # (σ33)I = (σ33)II
#     Eq5 = sp.Eq(λ1*(u0[0]*n[0]/v0 + u_refl_p[0]*nReflP[0]/vp1*Rp + u_refl_s1[0]*nReflS1[0]/vs1*Rs1 + u_refl_s2[0]*nReflS2[0]/vs1*Rs2 +\
#                     u0[1]*n[1]/v0 + u_refl_p[1]*nReflP[1]/vp1*Rp + u_refl_s1[1]*nReflS1[1]/vs1*Rs1 + u_refl_s2[1]*nReflS2[1]/vs1*Rs2) +\
#                 (λ1 + 2*μ1)*(u0[2]*n[2]/v0 + u_refl_p[2]*nReflP[2]/vp1*Rp + u_refl_s1[2]*nReflS1[2]/vs1*Rs1 + u_refl_s2[2]*nReflS2[2]/vs1*Rs2),\
#                 λ2*(u_trans_p[0]*nTransP[0]/vp2*Tp + u_trans_s1[0]*nTransS1[0]/vs2*Ts1 + u_trans_s2[0]*nTransS2[0]/vs2*Ts2 +\
#                     u_trans_p[1]*nTransP[1]/vp2*Tp + u_trans_s1[1]*nTransS1[1]/vs2*Ts1 + u_trans_s2[1]*nTransS2[1]/vs2*Ts2) +\
#                 (λ2 + 2*μ2)*(u_trans_p[2]*nTransP[2]/vp2*Tp + u_trans_s1[2]*nTransS1[2]/vs2*Ts1 + u_trans_s2[2]*nTransS2[2]/vs2*Ts2))
    
# #     Система собрана, сталось её решить. Сделаем это:
#     Solutions = sp.linsolve([Eq0,Eq1,Eq2,Eq3,Eq4,Eq5],[Rp,Rs1,Rs2,Tp,Ts1,Ts2])
    
# #     И "вытащим" искомые коэффициенты:
#     Rp = next(iter(Solutions))[0]
#     Rs1 = next(iter(Solutions))[1]
#     Rs2 = next(iter(Solutions))[2]    
#     Tp = next(iter(Solutions))[3]
#     Ts1 = next(iter(Solutions))[4]
#     Ts2 = next(iter(Solutions))[5]
  
#     Код выше закомментирован, т.к. он задействет библиотеку sympy, что не оправдано для решения задачи.
# Однако комментарии оттуда остаются полезными, да и построение системы там выглядит прозрачнее.
# Поэтому эта часть не удалена.

#     Зададим матрицу составленной системы:
# Первые три уравнения задаются легко. А вот оставшиеся три мы будем "вбивать" её не напрямую, а по алгоритму,
# представленому в "Лучевой метод в анизотропной среде (алгоритмы, программы)" Оболенцева, Гречки на стр. 97.
# Т.е. будем задавать коэффициенты системы через довольно хитрые циклы.
# Кроме всего прочего, этот алгоритм, вроде бы, универсален и для изотропных, и для анизотропных сред.

#     В этих циклах имеет место хитрое индексирование. И реализовать его можно, введя вспомогательную матрицу:
    matrix__of__indices = np.array([[0, 5, 4],
                                    [5, 1, 3],
                                    [4, 3, 2]])
#     Сами коэффициенты,которые мы хотим посчитать, будем записывать в отдельные массивы по три в каждый.
# Эти числа могут быть и комплексными, что надо указать при задании массивов.
# Падающая волна:
    u0__system__coefficients = np.zeros(3, dtype=complex)  # коэффициенты для падающей волны
# Отражённые волны:
    rp__system__coefficients = np.zeros(3, dtype=complex)  # коэффициенты для отражённой P-волны
    rs1__system__coefficients = np.zeros(3, dtype=complex)  # коэффициенты для отражённой S1-волны
    rs2__system__coefficients = np.zeros(3, dtype=complex)  # коэффициенты для отражённой S2-волны
# Преломлённые волны:
    tp__system__coefficients = np.zeros(3, dtype=complex)  # коэффициенты для преломлённой P-волны
    ts1__system__coefficients = np.zeros(3, dtype=complex)  # коэффициенты для преломлённой S1-волны
    ts2__system__coefficients = np.zeros(3, dtype=complex)  # коэффициенты для преломлённой S2-волны
    
#     Переходим к заполнению этих новых массивов. В каждом из них по три компоненты: по одной на каждое уравнение.
    for k in range(3):
        for i in range(3):
            for j in range(3):
                u0__system__coefficients[k] = u0__system__coefficients[k] +\
                    c_ij1[matrix__of__indices[k, 2], matrix__of__indices[i, j]] * p0[i] * u0[j]
                rp__system__coefficients[k] = rp__system__coefficients[k] +\
                    c_ij1[matrix__of__indices[k, 2], matrix__of__indices[i, j]] * p_refl_p[i] * u_refl_p[j]
                
                rs1__system__coefficients[k] = rs1__system__coefficients[k] +\
                    c_ij1[matrix__of__indices[k, 2], matrix__of__indices[i, j]]*p_refl_s1[i]*u_refl_s1[j]
                
                rs2__system__coefficients[k] = rs2__system__coefficients[k] +\
                    c_ij1[matrix__of__indices[k, 2], matrix__of__indices[i, j]]*p_refl_s2[i]*u_refl_s2[j]
                
                tp__system__coefficients[k] = tp__system__coefficients[k] +\
                    c_ij2[matrix__of__indices[k, 2], matrix__of__indices[i, j]]*p_trans_p[i]*u_trans_p[j]
                
                ts1__system__coefficients[k] = ts1__system__coefficients[k] +\
                    c_ij2[matrix__of__indices[k, 2], matrix__of__indices[i, j]]*p_trans_s1[i]*u_trans_s1[j]
                
                ts2__system__coefficients[k] = ts2__system__coefficients[k] +\
                    c_ij2[matrix__of__indices[k, 2], matrix__of__indices[i, j]]*p_trans_s2[i]*u_trans_s2[j]

    matrix__of__the__system = np.array([[u_refl_p[0],
                                         u_refl_s1[0],
                                         u_refl_s2[0],
                                         -u_trans_p[0],
                                         -u_trans_s1[0],
                                         -u_trans_s2[0]],

                                        [u_refl_p[1],
                                         u_refl_s1[1],
                                         u_refl_s2[1],
                                         -u_trans_p[1],
                                         -u_trans_s1[1],
                                         -u_trans_s2[1]],

                                        [u_refl_p[2],
                                         u_refl_s1[2],
                                         u_refl_s2[2],
                                         -u_trans_p[2],
                                         -u_trans_s1[2],
                                         -u_trans_s2[2]],

                                        [rp__system__coefficients[0],
                                         rs1__system__coefficients[0],
                                         rs2__system__coefficients[0],
                                         -tp__system__coefficients[0],
                                         -ts1__system__coefficients[0],
                                         -ts2__system__coefficients[0]],

                                        [rp__system__coefficients[1],
                                         rs1__system__coefficients[1],
                                         rs2__system__coefficients[1],
                                         -tp__system__coefficients[1],
                                         -ts1__system__coefficients[1],
                                         -ts2__system__coefficients[1]],

                                        [rp__system__coefficients[2],
                                         rs1__system__coefficients[2],
                                         rs2__system__coefficients[2],
                                         -tp__system__coefficients[2],
                                         -ts1__system__coefficients[2],
                                         -ts2__system__coefficients[2]]]
                                       )

#     Введём правую часть системы уравнений:
    right_part_of_the_system = np.array([-u0[0],
                                         -u0[1],
                                         -u0[2],
                                         -u0__system__coefficients[0],
                                         -u0__system__coefficients[1],
                                         -u0__system__coefficients[2]])
    
#     Решим эту систему уравнений:
    rp, rs1, rs2, tp, ts1, ts2 = np.linalg.solve(matrix__of__the__system, right_part_of_the_system)
    
#     Завершаем нашу функцию.
    return rp, rs1, rs2, tp, ts1, ts2

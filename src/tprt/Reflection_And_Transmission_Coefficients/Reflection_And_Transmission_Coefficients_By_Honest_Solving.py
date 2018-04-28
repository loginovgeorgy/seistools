# В этом скрипте будет реализвана функция, честно считающая коэффициенты отражения/прохождения на границе раздела двух упругих сред. Под "честным" здесь понимается составление системы уравнений на основе граничных условий и её решение.

# Система координат: ось X направлена вправо, ось Z - вниз, а ось Y - в плоскость экрана на нас. Модель "классическая": волна падает на горизонтальную границу двух ИЗОТРОПНЫХ сред в плоскости XZ под некоторым углом падения, отсчитываемым против часовой стрелки от нормали "вверх" от границы раздела. Поляризации волн могут быть трёх видов: продольная (сонаправлена с нормалью к фронту), поперечная-1 (отклонена на 90 градусов по часовой стрелке от нормали к фронту, лежит в плоскости экрана) и поперечная-2 (направлена в плоскость экрана на нас). Таким образом, тройка поляризаций "поперечная-2, поперечная-1 и продольная" - правая.

# Названия "S1" и "S2" означаеют соответственно "SV" и "SH" (комментарии в большинсвте своём писались в ходе работы, и поэтому было легче
# пользоваться индексацией, применяемой в коде.

# На вход функции будут подаваться данные о средах над и под границей раздела, как то: плотности, скорости продольных и поперечных волн. Кроме того, в аргументах функции необходимо указать тип падающей волны и угол её падения. Аргументы подаются в следующем порядке:

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

# ЕСТЬ СЛОЖНОСТЬ: ГДЕ-ТО ИСПОЛЬЗУЕТСЯ numpy (только действительные числа), А ГДЕ-ТО cmath (комплекснозначные функции). ВОЗМОЖНЫ ПРОБЛЕМЫ.

import numpy as np
import cmath as cm

from Cij_Matrix import *
from Waves_Velocities_And_Polarizations_In_Isotropic_Case import *

def Reflection_And_Transmission_Coefficients_By_Honest_Solving(σ1,σ2,Vp1,Vs1,Vp2,Vs2,Wave_Type,Angle_Deg):
    
#     для начала, нужно немного поколдовать над входными данными - перевести угол из градусов в радианы:
    Angle_Rad = np.radians(Angle_Deg)
    
    
#     далее, для решения поставленной задачи потребуются матрицы упругих модулей сред 1 и 2:
    Cij1 = Cij([Vp1,Vs1],σ1)
    Cij2 = Cij([Vp2,Vs2],σ2)
    
    
#     сформируем данные о падающей волне, найдём нормаль к её фронту, её поляризацию и вектор медленности:
    n = np.array([np.sin(Angle_Rad),0,np.cos(Angle_Rad)]) #нормаль к фронту

    if Wave_Type == 0:
        V = Vp1
    else:
        V = Vs1
        
    U = Polarizations_By_Christoffel_Equation(Cij1,σ1,n)
    #мы задали скорости волн в 1-й среде и их поляризации. Обозначения V,U будут часто
    #использоваться в данном скрипте для разных величин, так что не следует запоминать за ними конкретный физический смысл

    V0 = V #запомним скорость падающей волны
    U0 = U[:,Wave_Type]
    p0 = n/V0 #и её вектор медленности
    
    
#     найдём параметры отражённых и преломлённых волн (соответственно, Refl и Trans):
   
# вектора медленности:
    # отражённые волны
    pReflP = np.array([p0[0],p0[1],-cm.sqrt(1/Vp1**2 - p0[0]**2 - p0[1]**2)]) #первые две координаты сохраняются по закону Снеллиуса
    pReflS1 = np.array([p0[0],p0[1],-cm.sqrt(1/Vs1**2 - p0[0]**2 - p0[1]**2)]) #первые две координаты сохраняются по закону Снеллиуса
    pReflS2 = pReflS1 #векторы медленности S1 и S2 волн в изотропной среде совпадают
    # преломлённые волны
    pTransP = np.array([p0[0],p0[1],cm.sqrt(1/Vp2**2 - p0[0]**2 - p0[1]**2)]) #первые две координаты сохраняются по закону Снеллиуса
    pTransS1 = np.array([p0[0],p0[1],cm.sqrt(1/Vs2**2 - p0[0]**2 - p0[1]**2)]) #первые две координаты сохраняются по закону Снеллиуса
    pTransS2 = pTransS1 #векторы медленности S1 и S2 волн в изотропной среде совпадают
        
# поляризации отражённых и преломлённых волн:
    # отражённые волны
    
    U = Polarizations_By_Christoffel_Equation(Cij1,σ1,pReflP) #находим, с какими поляризациями волна может распространяться
    #в 1-й среде при заданном веторе медленности pReflP
    
    UReflP = U[:,0] #по построению, поляризация продольной волны - "первая в списке".
    
    U = Polarizations_By_Christoffel_Equation(Cij1,σ1,pReflS1)
    
    UReflS1 = U[:,1]
    UReflS2 = U[:,2] #считать отдельно матрицы V и U для волны S2 бессмысленно, т.к. нормаль к её фронту
#     совпадает с нормалью к фронту волны S1

    # преломлённые волны
    
    U = Polarizations_By_Christoffel_Equation(Cij2,σ2,pTransP) #находим, с какими поляризациями волна может распространяться
    #в 1-й среде при заданном веторе медленности pReflP
    
    UTransP = U[:,0] #по построению, поляризация продольной волны - "первая в списке".
    
    U = Polarizations_By_Christoffel_Equation(Cij2,σ2,pTransS1)
    
    UTransS1 = U[:,1]
    UTransS2 = U[:,2] #считать отдельно матрицы V и U для волны S2 бессмысленно, т.к. нормаль к её фронту
#     совпадает с нормалью к фронту волны S1

    # отражённые волны
    
#     переходим к составлению системы уравнений

# Нужно определить символы, относительно которых будем решать уравнния
# ЗАКОММЕНТИРОВАНО, Т.К. ИСПОЛЬЗОВАНИЕ БИБЛИОТЕКИ SYMPY ЗДЕСЬ ИЗЛИШНЕ!!!

#     Rp,Rs1,Rs2,Tp,Ts1,Ts2 = sp.symbols(' Rp,Rs1,Rs2,Tp,Ts1,Ts2 ')
    
    # нам понадобятся коэффициенты Ламе

#     λ1 = Cij1[0,0] - 2*Cij1[3,3]
#     μ1 = Cij1[3,3]

#     λ2 = Cij2[0,0] - 2*Cij2[3,3]
#     μ2 = Cij2[3,3]
    
# #     условие №1: нулевой скачок смещений на границе:
#     #приравниваем x-координаты смещений сверху и снизу от границы:
#     Eq0 = sp.Eq(U0[0] + UReflP[0]*Rp + UReflS1[0]*Rs1 + UReflS2[0]*Rs2,UTransP[0]*Tp + UTransS1[0]*Ts1 + UTransS2[0]*Ts2)
#     #приравниваем y-координаты смещений сверху и снизу от границы:
#     Eq1 = sp.Eq(U0[1] + UReflP[1]*Rp + UReflS1[1]*Rs1 + UReflS2[1]*Rs2,UTransP[1]*Tp + UTransS1[1]*Ts1 + UTransS2[1]*Ts2)
#     #приравниваем z-координаты смещений сверху и снизу от границы:
#     Eq2 = sp.Eq(U0[2] + UReflP[2]*Rp + UReflS1[2]*Rs1 + UReflS2[2]*Rs2,UTransP[2]*Tp + UTransS1[2]*Ts1 + UTransS2[2]*Ts2)
    
#     # условие №2: нулевой скачок нормальных напряжений на границе
# # с помощью закона Гука и приближения малых деформаций это условие сводится к условиям на производные вектора смещений;
# # далее используется предположение о виде сигнала
#     # (σ13)I = (σ13)II
#     Eq3 = sp.Eq(μ1*(U0[0]*n[2]/V0 + UReflP[0]*nReflP[2]/Vp1*Rp + UReflS1[0]*nReflS1[2]/Vs1*Rs1 + UReflS2[0]*nReflS2[2]/Vs1*Rs2 +\
#                     U0[2]*n[0]/V0 + UReflP[2]*nReflP[0]/Vp1*Rp + UReflS1[2]*nReflS1[0]/Vs1*Rs1 + UReflS2[2]*nReflS2[0]/Vs1*Rs2),\
#                 μ2*(UTransP[0]*nTransP[2]/Vp2*Tp + UTransS1[0]*nTransS1[2]/Vs2*Ts1 + UTransS2[0]*nTransS2[2]/Vs2*Ts2 +\
#                     UTransP[2]*nTransP[0]/Vp2*Tp + UTransS1[2]*nTransS1[0]/Vs2*Ts1 + UTransS2[2]*nTransS2[0]/Vs2*Ts2))
#     # (σ23)I = (σ23)II
#     Eq4 = sp.Eq(μ1*(U0[1]*n[2]/V0 + UReflP[1]*nReflP[2]/Vp1*Rp + UReflS1[1]*nReflS1[2]/Vs1*Rs1 + UReflS2[1]*nReflS2[2]/Vs1*Rs2 +\
#                     U0[2]*n[1]/V0 + UReflP[2]*nReflP[1]/Vp1*Rp + UReflS1[2]*nReflS1[1]/Vs1*Rs1 + UReflS2[2]*nReflS2[1]/Vs1*Rs2),\
#                 μ2*(UTransP[1]*nTransP[2]/Vp2*Tp + UTransS1[1]*nTransS1[2]/Vs2*Ts1 + UTransS2[1]*nTransS2[2]/Vs2*Ts2 +\
#                     UTransP[2]*nTransP[1]/Vp2*Tp + UTransS1[2]*nTransS1[1]/Vs2*Ts1 + UTransS2[2]*nTransS2[1]/Vs2*Ts2))
#     # (σ33)I = (σ33)II
#     Eq5 = sp.Eq(λ1*(U0[0]*n[0]/V0 + UReflP[0]*nReflP[0]/Vp1*Rp + UReflS1[0]*nReflS1[0]/Vs1*Rs1 + UReflS2[0]*nReflS2[0]/Vs1*Rs2 +\
#                     U0[1]*n[1]/V0 + UReflP[1]*nReflP[1]/Vp1*Rp + UReflS1[1]*nReflS1[1]/Vs1*Rs1 + UReflS2[1]*nReflS2[1]/Vs1*Rs2) +\
#                 (λ1 + 2*μ1)*(U0[2]*n[2]/V0 + UReflP[2]*nReflP[2]/Vp1*Rp + UReflS1[2]*nReflS1[2]/Vs1*Rs1 + UReflS2[2]*nReflS2[2]/Vs1*Rs2),\
#                 λ2*(UTransP[0]*nTransP[0]/Vp2*Tp + UTransS1[0]*nTransS1[0]/Vs2*Ts1 + UTransS2[0]*nTransS2[0]/Vs2*Ts2 +\
#                     UTransP[1]*nTransP[1]/Vp2*Tp + UTransS1[1]*nTransS1[1]/Vs2*Ts1 + UTransS2[1]*nTransS2[1]/Vs2*Ts2) +\
#                 (λ2 + 2*μ2)*(UTransP[2]*nTransP[2]/Vp2*Tp + UTransS1[2]*nTransS1[2]/Vs2*Ts1 + UTransS2[2]*nTransS2[2]/Vs2*Ts2))
    
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
# Однако комментарии оттуда остаются полезными, да и построение системы там выглядит прозрачнее. Поэтому эта часть не удалена.
# Стоит отметить, что numpy.linalg умеет обращаться с комплексными числами.

#     Зададим матрицу составленной системы:
# Первые три уравнения задаются легко. А вот оставшиеся три мы будем "вбивать" её не напрямую, а по алгоритму,
# представленому в "Лучевой метод в анизотропной среде (алгоритмы, программы)" Оболенцева, Гречки.
# Т.е. будем задавать коэффициенты системы через довольно хитрые циклы.
# Кроме всего прочего, этот алгоритм, вроде бы, универсален и для изотропных, и для анизотропных сред.

#     В этих циклах имеет место хитрое индексирование. И реализовать его можно, введя вспомогательную матрицу:
    Matrix_Of_Indices = np.array([[0, 5, 4],
                                  [5, 1, 3],
                                  [4, 3, 2]])
#     Сами коэффициенты,которые мы хотим посчитать, будем записывать в отдельные массивы по три в каждый.
# Эти числа могут быть и комплексными, что надо указать при задании массивов.
# Падающая волна:
    U0_System_Coefficients = np.zeros(3,dtype = complex) #коэффициенты для падающей волны
# Отражённые волны:
    Rp_System_Coefficients = np.zeros(3,dtype = complex) #коэффициенты для отражённой P-волны
    Rs1_System_Coefficients = np.zeros(3,dtype = complex) #коэффициенты для отражённой S1-волны
    Rs2_System_Coefficients = np.zeros(3,dtype = complex) #коэффициенты для отражённой S2-волны
# Преломлённые волны:
    Tp_System_Coefficients = np.zeros(3,dtype = complex) #коэффициенты для преломлённой P-волны
    Ts1_System_Coefficients = np.zeros(3,dtype = complex) #коэффициенты для преломлённой S1-волны
    Ts2_System_Coefficients = np.zeros(3,dtype = complex) #коэффициенты для преломлённой S2-волны
    
#     Переходим к заполнению этих новых массивов. В каждом из них по три компоненты: по одной на каждое уравнение.
    for k in range(3):
        for i in range(3):
            for j in range(3):
                U0_System_Coefficients[k] = U0_System_Coefficients[k] +\
                Cij1[Matrix_Of_Indices[k,2],Matrix_Of_Indices[i,j]]*p0[i]*U0[j]
                
                Rp_System_Coefficients[k] = Rp_System_Coefficients[k] +\
                Cij1[Matrix_Of_Indices[k,2],Matrix_Of_Indices[i,j]]*pReflP[i]*UReflP[j]
                
                Rs1_System_Coefficients[k] = Rs1_System_Coefficients[k] +\
                Cij1[Matrix_Of_Indices[k,2],Matrix_Of_Indices[i,j]]*pReflS1[i]*UReflS1[j]
                
                Rs2_System_Coefficients[k] = Rs2_System_Coefficients[k] +\
                Cij1[Matrix_Of_Indices[k,2],Matrix_Of_Indices[i,j]]*pReflS2[i]*UReflS2[j]
                
                Tp_System_Coefficients[k] = Tp_System_Coefficients[k] +\
                Cij2[Matrix_Of_Indices[k,2],Matrix_Of_Indices[i,j]]*pTransP[i]*UTransP[j]
                
                Ts1_System_Coefficients[k] = Ts1_System_Coefficients[k] +\
                Cij2[Matrix_Of_Indices[k,2],Matrix_Of_Indices[i,j]]*pTransS1[i]*UTransS1[j]
                
                Ts2_System_Coefficients[k] = Ts2_System_Coefficients[k] +\
                Cij2[Matrix_Of_Indices[k,2],Matrix_Of_Indices[i,j]]*pTransS2[i]*UTransS2[j]
    

    Matrix_Of_The_System = np.array([[UReflP[0],UReflS1[0],UReflS2[0],-UTransP[0],-UTransS1[0],-UTransS2[0]],
                      [UReflP[1],UReflS1[1],UReflS2[1],-UTransP[1],-UTransS1[1],-UTransS2[1]],
                      [UReflP[2],UReflS1[2],UReflS2[2],-UTransP[2],-UTransS1[2],-UTransS2[2]],
                      [Rp_System_Coefficients[0],Rs1_System_Coefficients[0],Rs2_System_Coefficients[0],\
                       -Tp_System_Coefficients[0],-Ts1_System_Coefficients[0],-Ts2_System_Coefficients[0]],
                      [Rp_System_Coefficients[1],Rs1_System_Coefficients[1],Rs2_System_Coefficients[1],\
                       -Tp_System_Coefficients[1],-Ts1_System_Coefficients[1],-Ts2_System_Coefficients[1]],
                      [Rp_System_Coefficients[2],Rs1_System_Coefficients[2],Rs2_System_Coefficients[2],\
                       -Tp_System_Coefficients[2],-Ts1_System_Coefficients[2],-Ts2_System_Coefficients[2]]])

#     Введём правую часть системы уравнений:
    Right_Part_Of_The_System = np.array([-U0[0],
                               -U0[1],
                               -U0[2],
                               -U0_System_Coefficients[0],
                               -U0_System_Coefficients[1],
                               -U0_System_Coefficients[2]])
    
#     Решим эту систему уравнений:
    Rp,Rs1,Rs2,Tp,Ts1,Ts2 = np.linalg.solve(Matrix_Of_The_System,Right_Part_Of_The_System)
    
#     Завершаем нашу функцию.
    return Rp,Rs1,Rs2,Tp,Ts1,Ts2
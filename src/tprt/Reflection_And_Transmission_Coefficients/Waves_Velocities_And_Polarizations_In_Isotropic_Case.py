import numpy as np
import cmath as cm

from .Cij_Matrix import *

# Обе функции написаны для 2,5-D модели: вектора и пространство трёхмерные, но вдоль оси Y ничего не меняется. Вектор нормали к фронту падающей волны лежит в плоскости XZ. Ориентация осей следующая: ось X направлена вправо вдоль экрана, ось Z - вниз вдоль экрана. Ось Y смотрит в экран на нас. Среда ИЗОТРОПНАЯ.

# на вход нижеописанной функции подаются:

# Cij - 6x6-матрица упругих модулей среды
# Density - скалярная величина, плотность среды
# n - 3x1-матрица (т.е. вектор из трёх координат) - единичный вектор нормали к фронту волны

# на выхоод нижеописанной функции будет:

# V - 3x1-матрица (т.е. вектор из трёх координат) - вектор скоростей упругих волн в среде
# скорости в этом векторе будут отсортированы в следующем порядке: (Vp,Vs1,Vs2)
# Стоит иметь ввиду, что в изотропной среде скорости продольных волн Vs1 и Vs2 совпадают
# U - 3x3-матрица - матрица, столбцами которой будут вектора поляризации упругих волн в среде
# Поляризации поперечных волн будут таковы: первая из них лежит в плосоксти экрана, а вторая смотрит в экран на нас. Тройка Us2,Us1,Up - правая.

# НАДО НАПИСАТЬ ПРОВЕРКИ, В Т.Ч. НА ЕДИНИЧНОСТЬ ВЕКТОРА n И НА ЕГО ПРИНАДЛЕЖНОСТЬ ПЛОСКОСТИ OXZ.

def PhaseVel_and_Polariz(Cij,Density,n):
    
#     задаём формат вывода
    V = np.array(0,dtype = complex)
    V.resize(3)
    U = np.array(0,dtype = complex)
    U.resize(3,3)
    
#     находим искомые вектор и матрицу

# скорости находятся из матрицы упругих модулей
    V[0] = cm.sqrt(Cij[0,0]/Density)
    V[1] = cm.sqrt(Cij[3,3]/Density)
    V[2] = cm.sqrt(Cij[3,3]/Density) #для изотропной среды скорости обеих поперечных волн совпадают
    
#     поляризация продольной волны совпадает с направлением распространения волны
    U[:,0] = n #поляризация продольной волны
    
#     поляризацию поперечной волны можно задать по-разному, но мы зададим так:
    U[:,1] = ([-n[2],0,n[0]]) #поляризация "первой" поперечной волны
    U[:,2] = ([0,1,0]) #поляризация "второй" поперечной волны    
    
    return V,U

# Следующая функция будет честно, по заданной матрице упругих модулей среды Cij, плотности среды и направлению в пространстве,
# будет рассчитывать векторы поляризации волн, которые могут распространяться в данном направлении (все вышеуказанные условия остаются в силе).

# на вход этой функции подаются аргументы в следующем порядке:

# Cij - 6x6-матрица упругих модулей среды
# Density - скалярная величина, плотность среды
# n - 3x1-матрица (т.е. вектор из трёх координат) - направление в пространстве

# на выхоод этой функции будет:

# U - 3x3-матрица - матрица, столбцами которой будут вектора поляризации упругих волн в среде

# Поляризации поперечных волн будут таковы: первая из них лежит в плосоксти экрана, а вторая смотрит в экран на нас. Тройка Us2,Us1,Up - правая.

def Polarizations_By_Christoffel_Equation(Cij,Density,p):
    
#     задаём формат вывода
    U = np.array(0,dtype = complex)
    U.resize(3,3)
    
#     находим весь тензор Cijkl:
    C_ijkl = Cijkl(Cij)
#     почему-то тут надо использовать форму Cijkl*Density:
    C_ijkl = C_ijkl/Density

#     Строим матрицу Кристоффеля Гik = Cijkl*pj*pl:
    Гik = np.array(0,dtype = complex)
    Гik.resize(3,3)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    Гik[i,k] = Гik[i,k] + C_ijkl[i,j,k,l]*p[j]*p[l]
                    
#     И находим собственные векторы этой матрицы. Они и будут векторами поляризации.
# Однако эти векторы будут в невесть каком порядке. Исправим это: отсортируем соственные числа по возрастанию:
    Eigenvalues = np.linalg.eig(Гik)[0]
    I = np.argsort(Eigenvalues) #запоминаем перестановку
    
    Polarizations = np.linalg.eig(Гik)[1] #отсюда будем брать векторы поляризвации.
    
#     поляризация продольной волны совпадает с направлением распространения волны
    U[:,0] = Polarizations[:,I[2]] #поляризация продольной волны
    if U[:,0].dot(p) < 0:
        U[:,0] = - U[:,0] #Продольная волна должна быть поляризована в направлении распространения.
    
#     поляризацию поперечной волны можно задать по-разному, но мы зададим так:
    for i in range(2):
        if Polarizations[:,I[i]][1] == 0: #если Y-координата собственного вектора, соответствующего второму собственному числу, равна 0...
            U[:,1] = Polarizations[:,I[1]] #поляризация "первой" поперечной волны.
            U[:,2] = Polarizations[:,I[0]] #поляризация "второй" поперечной волны   
            
        else: #если же нет...
            U[:,1] = Polarizations[:,I[0]] #поляризация "первой" поперечной волны.
            #"Минус" - чтобы выполнить наши соглашения об ориентации векторов.
            U[:,2] = Polarizations[:,I[1]] #поляризация "второй" поперечной волны    

    
    return U
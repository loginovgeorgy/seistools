import numpy as np

# среда ИЗОТРОПНАЯ

# на вход нижеописанной функции подаются:
# Velocities - 1x2-вектор скоростей в следующием порядке: [Vp,Vs1]
# Density - скалярная величина, плотность среды

# на выходе будет:
# Cij - 6x6-матрица упругих модулей среды


def c_ij(velocities, density):
    
    # в изотропном случае в данной матрице будет всего два независимых элемента. Однако для удобсвта введём три:
    c00 = density*(velocities[0]**2)
    c33 = density*(velocities[1]**2)
    c01 = c00 - 2*c33
    
    # заполняем матрицу:

    c_ij = np.zeros((6, 6))

    c_ij[0, 0] = c00
    c_ij[1, 1] = c00
    c_ij[2, 2] = c00

    c_ij[3, 3] = c33
    c_ij[4, 4] = c33
    c_ij[5, 5] = c33

    c_ij[0, 1] = c01
    c_ij[0, 2] = c01
    c_ij[1, 2] = c01
    c_ij[1, 0] = c01
    c_ij[2, 0] = c01
    c_ij[2, 1] = c01
    
    return c_ij

# Следующей функции на вход подаётся матрица Cij, а на выходе будет 3x3x3x3-тензор упругих модулей среды Cijkl
# Используется так называемая нотация Фойгта, согласно которой элементы тензора Cijkl связаны с элементами матрицы Cij
# следующими соотношениями: (слева - пары ij или kl из Cijkl, а справа - индескы i или j из Cij)

# 11 -> 1; 22 -> 2; 33 -> 3;
# 23,32 -> 4; 13,31 -> 5; 12,21 -> 6.

# НО: в "Питоне" индексы начинаются с нуля, а НЕ с единицы.
# Поэтому наша форма этой нотации будет несколько отличатся от заданной выше: у всех чисел нужно отнять по единице.


# Поэтому сначала зададим эти переходы как функцию индексов i и j:
def voigt_notation(i, j):
    if i == j:
        return i
    else:
        if [i, j] == [1, 2] or [i, j] == [2, 1]:
            return 3
        if [i, j] == [0, 2] or [i, j] == [2, 0]:
            return 4
        if [i, j] == [0, 1] or [i, j] == [1, 0]:
            return 5


# А теперь перйдём к восстановлению тензора Cijkl:
def c_ijkl(c_ij):
    
    c_ijkl = np.zeros((3, 3, 3, 3))
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    
                    from_i = voigt_notation(i, j)
                    from_j = voigt_notation(k, l)
                    
                    c_ijkl[i, j, k, l] = c_ij[from_i, from_j]
                    
    return c_ijkl

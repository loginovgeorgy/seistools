import numpy as np

# среда ИЗОТРОПНАЯ

# на вход нижеописанной функции подаются:
# velocities - 1x2-вектор скоростей в следующием порядке: [vp,vs]
# density - скалярная величина, плотность среды

# на выходе будет:
# c_ij - 6x6-матрица упругих модулей среды


def c_ij(velocities, density):
    
    # в изотропном случае в данной матрице будет всего два независимых элемента.
    # заполняем матрицу:

    c_ij1 = np.zeros((6, 6)) # index "1" beside "c_ij" marks that this is a variable,
    # not a function defined above

    c_ij1[0, 0] = density*(velocities[0]**2)
    c_ij1[1, 1] = c_ij1[0, 0]
    c_ij1[2, 2] = c_ij1[0, 0]

    c_ij1[3, 3] = density*(velocities[1]**2)
    c_ij1[4, 4] = c_ij1[3, 3]
    c_ij1[5, 5] = c_ij1[3, 3]

    c_ij1[0, 1] = c_ij1[0, 0] - 2 * c_ij1[3, 3]
    c_ij1[0, 2] = c_ij1[0, 1]
    c_ij1[1, 2] = c_ij1[0, 1]
    c_ij1[1, 0] = c_ij1[0, 1]
    c_ij1[2, 0] = c_ij1[0, 1]
    c_ij1[2, 1] = c_ij1[0, 1]
    
    return c_ij1

# Следующей функции на вход подаётся матрица c_ij, а на выходе будет 3x3x3x3-тензор упругих модулей среды c_ijkl
# Используется так называемая нотация Фойгта, согласно которой элементы тензора c_ijkl связаны с элементами матрицы c_ij
# следующими соотношениями: (слева - пары ij или kl из c_ijkl, а справа - индескы i или j из c_ij)

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


# А теперь перйдём к восстановлению тензора c_ijkl:
def c_ijkl(c_ij1): # index "1" beside "c_ij" marks that this is a variable,
    # not a function defined above
    
    c_ijkl1 = np.zeros((3, 3, 3, 3)) # the same index with the same purpose
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    
                    from_i = voigt_notation(i, j)
                    from_j = voigt_notation(k, l)
                    
                    c_ijkl1[i, j, k, l] = c_ij1[from_i, from_j]
                    
    return c_ijkl1

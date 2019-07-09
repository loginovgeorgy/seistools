import numpy as np
import cmath as cm

from .c_ij_matrix import iso_c_ij, voigt_notation, c_ijkl_from_c_ij
from .polarizations import christoffel, polarizations, polarizations_alt


def _normalize_vector(v):
    """Makes sum of squared components of a vector be equal to one

    :param v: vector of any dimensions
    :return: vector v divided by square root of the sum of its squared components
    """
    return v / cm.sqrt(np.sum(v ** 2))


def iso_rt_coefficients(inc_slowness, inc_polariz, rt_signum, vp1, vs1, rho1, vp2, vs2, rho2):
    """Computes reflection / transmission coefficients on a boundary connecting two isotropic media.

    :param inc_slowness: slowness vector of the incident wave in local coordinate system
    :param inc_polariz: polarization vector of the incident wave in local coordinate system
    :param rt_signum: desired output's flag: + 1 means transition, - 1 stands for reflection
    :param vp1: P-wave velocity in the overburden layer
    :param vs1: S-wave velocity in the overburden layer
    :param rho1: density in the overburden medium
    :param vp2: P-wave velocity in the underlying layer
    :param vs2: S-wave velocity in the underlying layer
    :param rho2: density in the underlying layer
    :return: array of reflection / transmission coefficients (depending on value of rt_signum)
    """

    # Check if incident slowness lies in vertical plane (i.e. it is given in local coordinates):
    if abs(inc_slowness[1] / np.linalg.norm(inc_slowness)) >= 0.01:
        # In local coordinates inc_slowness[1] must be equal to 0.
        raise Exception("Slowness of the incident wave does not lie in vertical plane.")

    # Construct full slownesses of reflected and transmitted waves.
    # Tangent component of the slowness is preserved, the remaining one can be computed using slowness's definition.
    # Note that we assume that in local coordinate system "y" component of incident slowness is equal to zero.

    refl_slow_p = np.array([inc_slowness[0],  # reflected P-wave slowness
                            0,
                            - cm.sqrt(1 / vp1 ** 2 - inc_slowness[0] ** 2)])
    refl_slow_s = np.array([inc_slowness[0],  # reflected S-waves slowness
                            0,
                            - cm.sqrt(1 / vs1 ** 2 - inc_slowness[0] ** 2)])

    trans_slow_p = np.array([inc_slowness[0],  # transmitted P-wave slowness
                             0,
                             cm.sqrt(1 / vp2 ** 2 - inc_slowness[0] ** 2)])
    trans_slow_s = np.array([inc_slowness[0],  # transmitted S-wave slowness
                             0,
                             cm.sqrt(1 / vs2 ** 2 - inc_slowness[0] ** 2)])

    # Construct polarizations of reflected and transmitted waves:

    refl_polariz_p = refl_slow_p / np.linalg.norm(refl_slow_p)  # reflected P-wave
    refl_polariz_s2 = np.array([0, 1, 0])  # reflected SH-wave
    refl_polariz_s1 = np.cross(refl_polariz_s2, refl_slow_s / np.linalg.norm(refl_slow_s))  # reflected SV-wave

    trans_polariz_p = trans_slow_p / np.linalg.norm(trans_slow_p)  # transmitted P-wave
    trans_polariz_s2 = np.array([0, 1, 0])  # transmitted SH-wave
    trans_polariz_s1 = np.cross(trans_polariz_s2, trans_slow_s / np.linalg.norm(trans_slow_s))  # transmitted SV-wave

    # They are unit in hermite norm. But in theory they should be unit in sense of sum of squared components:

    refl_polariz_p = _normalize_vector(refl_polariz_p)
    refl_polariz_s1 = _normalize_vector(refl_polariz_s1)
    refl_polariz_s2 = _normalize_vector(refl_polariz_s2)

    trans_polariz_p = _normalize_vector(trans_polariz_p)
    trans_polariz_s1 = _normalize_vector(trans_polariz_s1)
    trans_polariz_s2 = _normalize_vector(trans_polariz_s2)

    # All polarizations are subject to boundary conditions i.e. continuity of displacement and stress. These
    # conditions give raise to a system of linear equations with respect to some coefficients of proportionality. They
    # are known as reflection and transmission coefficients. Each polarization has enters the system with different
    # factors. Note that these conditions are written in general form for arbitrary anisotropic medium.

    # We'll need full stiffness tensors of both media. They will be constructed using their matrix form:

    c_ij1 = iso_c_ij(vp1, vs1, rho1 / 1000)  # we divide density by 1000 since numbers in c_ij are too big
    c_ij2 = iso_c_ij(vp2, vs2, rho2 / 1000)  # otherwise

    c_ijkl1 = c_ijkl_from_c_ij(c_ij1)
    c_ijkl2 = c_ijkl_from_c_ij(c_ij2)

    # Incident wave factors:
    inc_factors = np.einsum("ikl, k, l", c_ijkl1[:, 2, :, :], inc_slowness, _normalize_vector(inc_polariz))

    # Reflected wave factors:
    refl_factors_p = np.einsum("ikl, k, l", c_ijkl1[:, 2, :, :], refl_slow_p, refl_polariz_p)
    refl_factors_s1 = np.einsum("ikl, k, l", c_ijkl1[:, 2, :, :], refl_slow_s, refl_polariz_s1)
    refl_factors_s2 = np.einsum("ikl, k, l", c_ijkl1[:, 2, :, :], refl_slow_s, refl_polariz_s2)

    # Transmitted wave factors:
    trans_factors_p = np.einsum("ikl, k, l", c_ijkl2[:, 2, :, :], trans_slow_p, trans_polariz_p)
    trans_factors_s1 = np.einsum("ikl, k, l", c_ijkl2[:, 2, :, :], trans_slow_s, trans_polariz_s1)
    trans_factors_s2 = np.einsum("ikl, k, l", c_ijkl2[:, 2, :, :], trans_slow_s, trans_polariz_s2)

    # Form up matrix of the system:

    matrix = np.array([
        [
            refl_polariz_p[0],
            refl_polariz_s1[0],
            refl_polariz_s2[0],
            - trans_polariz_p[0],
            - trans_polariz_s1[0],
            - trans_polariz_s2[0]
        ],
        [
            refl_polariz_p[1],
            refl_polariz_s1[1],
            refl_polariz_s2[1],
            - trans_polariz_p[1],
            - trans_polariz_s1[1],
            - trans_polariz_s2[1]
        ],
        [
            refl_polariz_p[2],
            refl_polariz_s1[2],
            refl_polariz_s2[2],
            - trans_polariz_p[2],
            - trans_polariz_s1[2],
            - trans_polariz_s2[2]
        ],
        [
            refl_factors_p[0],
            refl_factors_s1[0],
            refl_factors_s2[0],
            - trans_factors_p[0],
            - trans_factors_s1[0],
            - trans_factors_s2[0]
        ],
        [
            refl_factors_p[1],
            refl_factors_s1[1],
            refl_factors_s2[1],
            - trans_factors_p[1],
            - trans_factors_s1[1],
            - trans_factors_s2[1]
        ],
        [
            refl_factors_p[2],
            refl_factors_s1[2],
            refl_factors_s2[2],
            - trans_factors_p[2],
            - trans_factors_s1[2],
            - trans_factors_s2[2]
        ]
    ])

    # And right part of the system:

    right_part = np.array([
        - inc_polariz[0] / np.linalg.norm(inc_polariz),
        - inc_polariz[1] / np.linalg.norm(inc_polariz),
        - inc_polariz[2] / np.linalg.norm(inc_polariz),
        - inc_factors[0],
        - inc_factors[1],
        - inc_factors[2]
    ])

    # Sought coefficients of proportionality are solutions of this system:

    rp, rs1, rs2, tp, ts1, ts2 = np.linalg.solve(matrix, right_part)

    # Output depends on value of rt_signum

    if rt_signum == 1:

        return tp, ts1, ts2

    else:

        return rp, rs1, rs2


# Define error that will occur if incident slowness in iso_rt_coefficietns(...) does not lie in vertical plane (e.g. it
# is given in global coordinates, not in local ones).
class IncidentSlownessError(Exception):
    """Exception raised for errors in the input."""

    def __init__(self, msg):
        self.message = msg

# def rt_coefficients_alt(vp1, vs1, rho1, vp2, vs2, rho2, cos_inc, inc_polariz, inc_vel, rt_signum):
#
#     # Для решения поставленной задачи потребуются матрицы упругих модулей сред 1 и 2:
#     c_ij1 = iso_c_ij([vp1, vs1], rho1 / 1000)
#     c_ij2 = iso_c_ij([vp2, vs2], rho2 / 1000)
#     # Делим на 1000, чтобы избежать зашкаливающе больших чисел и соответствующих ошибок. На решения системы такая
#     # нормировка не повлияет.
#
#     # сформируем данные о падающей волне, её волновой вектор k0, её поляризацию и вектор медленности:
#     k0 = np.array([np.sqrt(1 - cos_inc ** 2), 0, cos_inc])  # нормаль к фронту
#
#     v0 = inc_vel
#     u0 = inc_polariz / np.linalg.norm(inc_polariz) # нормируем, т.к. все остальные векторы будут единичными
#     p0 = k0 / v0  # и её вектор медленности
#
#     # Создадим два массива. В первом будут лежать волновые векторы-строки отражённых волн, а во втором -
#     # преломлённых.
#
#     k_refl = np.array([np.zeros(3, dtype = complex), # P-волна
#                        np.zeros(3, dtype = complex), # SV-волна
#                        np.zeros(3, dtype = complex)]) # SH-волна
#
#     k_trans = np.array([np.zeros(3, dtype = complex), # P-волна
#                         np.zeros(3, dtype = complex), # SV-волна
#                         np.zeros(3, dtype = complex)]) # SH-волна
#
#     # Зададим сначала все волновые векторы в соответствии с законом Снеллиуса:
#
#     k_refl[0] = np.array([k0[0] * vp1 / v0,
#                           0,
#                           - cm.sqrt(1 - (k0[0] * vp1 / v0) ** 2)])
#
#     k_refl[1] = np.array([k0[0] * vs1 / v0,
#                           0,
#                           - cm.sqrt(1 - (k0[0] * vs1 / v0) ** 2)])
#     k_refl[2] = k_refl[1]  # волновые векторы для SV- и SH-волн в изотропной среде совпадают
#
#     k_trans[0] = np.array([k0[0] * vp2 / v0,
#                            0,
#                            cm.sqrt(1 - (k0[0] * vp2 / v0) ** 2)])
#
#     k_trans[1] = np.array([k0[0] * vs2 / v0,
#                            0,
#                            cm.sqrt(1 - (k0[0] * vs2 / v0) ** 2)])
#     k_trans[2] = k_trans[1]  # волновые векторы для SV- и SH-волн в изотропной среде совпадают
#
#     # Теперь заводим векторы медленности:
#
#     # отражённые волны
#     p_refl_p = k_refl[0] / vp1
#
#     p_refl_s1 = k_refl[1] / vs1
#
#     p_refl_s2 = k_refl[2] / vs1
#
#     # преломлённые волны
#
#     p_trans_p = k_trans[0] / vp2
#
#     p_trans_s1 = k_trans[1] / vs2
#
#     p_trans_s2 = k_trans[2] / vs2
#
#     # Поляризации отражённых и преломлённых волн:
#     # отражённые волны
#
#     u = polarizations_alt(c_ij1, p_refl_p)  # находим, с какими поляризациями волна может распространяться
#     # в 1-й среде при заданном веторе медленности p_refl_p
#
#     u_refl_p = u[:, 0]  # по построению, поляризация продольной волны - "первая в списке".
#
#     u = polarizations_alt(c_ij1, p_refl_s1)
#
#     u_refl_s1 = u[:, 1]
#     u_refl_s2 = u[:, 2]  # считать отдельно матрицы v и u для волны S2 бессмысленно, т.к. нормаль к её фронту
#     # совпадает с нормалью к фронту волны S1
#
#     # преломлённые волны
#
#     u = polarizations_alt(c_ij2, p_trans_p)  # находим, с какими поляризациями волна может распространяться
#     # в 1-й среде при заданном веторе медленности p_refl_p
#
#     u_trans_p = u[:, 0]  # по построению, поляризация продольной волны - "первая в списке".
#
#     u = polarizations_alt(c_ij2, p_trans_s1)
#
#     u_trans_s1 = u[:, 1]
#     u_trans_s2 = u[:, 2]  # считать отдельно матрицы v и u для волны S2 бессмысленно, т.к. нормаль к её фронту
#     # совпадает с нормалью к фронту волны S1
#
#     # Зададим матрицу системы уравнений на границе.
#
#     # Первые три уравнения задаются легко. А вот оставшиеся три мы будем "вбивать" не напрямую, а по алгоритму,
#     # представленому в "Лучевой метод в анизотропной среде (алгоритмы, программы)" Оболенцева, Гречки на стр. 97.
#     # Т.е. будем задавать коэффициенты системы через довольно хитрые циклы.
#     # Кроме всего прочего, этот алгоритм, вроде бы, универсален и для изотропных, и для анизотропных сред.
#
#     # Коэффициенты в матрице могут быть и комплексными, что надо указать при задании массивов.
#
#     # Падающая волна:
#     u0_coeff = np.zeros(3, dtype = complex) #коэффициенты для падающей волны
#
#     # Отражённые волны:
#     rp_coeff = np.zeros(3, dtype = complex) #коэффициенты для отражённой P-волны
#     rs1_coeff = np.zeros(3, dtype = complex) #коэффициенты для отражённой S1-волны
#     rs2_coeff = np.zeros(3, dtype = complex) #коэффициенты для отражённой S2-волны
#
#     # Преломлённые волны:
#     tp_coeff = np.zeros(3, dtype = complex) #коэффициенты для преломлённой P-волны
#     ts1_coeff = np.zeros(3, dtype = complex) #коэффициенты для преломлённой S1-волны
#     ts2_coeff = np.zeros(3, dtype = complex) #коэффициенты для преломлённой S2-волны
#
#     # При задании системы понадобится полный тензор упругих модулей:
#
#     c_ijkl_1 = c_ijkl_from_c_ij(c_ij1)
#     c_ijkl_2 = c_ijkl_from_c_ij(c_ij2)
#
#     # Заполненяем в цикле векторы коэффициентов системы:
#
#     for i in range(3):
#         for j in range(3):
#             for k in range(3):
#
#                 u0_coeff[i] = u0_coeff[i] +\
#                               c_ijkl_1[i, 2, j, k] * p0[j] * u0[k]
#
#                 rp_coeff[i] = rp_coeff[i] +\
#                               c_ijkl_1[i, 2, j, k] * p_refl_p[j] * u_refl_p[k]
#
#                 rs1_coeff[i] = rs1_coeff[i] +\
#                                c_ijkl_1[i, 2, j, k] * p_refl_s1[j] * u_refl_s1[k]
#
#                 rs2_coeff[i] = rs2_coeff[i] +\
#                                c_ijkl_1[i, 2, j, k] * p_refl_s2[j] * u_refl_s2[k]
#
#
#
#                 tp_coeff[i] = tp_coeff[i] +\
#                               c_ijkl_2[i, 2, j, k] * p_trans_p[j] * u_trans_p[k]
#
#                 ts1_coeff[i] = ts1_coeff[i] +\
#                                c_ijkl_2[i, 2, j, k] * p_trans_s1[j] * u_trans_s1[k]
#
#                 ts2_coeff[i] = ts2_coeff[i] +\
#                                c_ijkl_2[i, 2, j, k] * p_trans_s2[j] * u_trans_s2[k]
#
#     # Итого:
#
#     matrix = np.array([[u_refl_p[0], u_refl_s1[0], u_refl_s2[0],
#                         - u_trans_p[0], - u_trans_s1[0], - u_trans_s2[0]],
#                        [u_refl_p[1], u_refl_s1[1], u_refl_s2[1],
#                         - u_trans_p[1], - u_trans_s1[1], - u_trans_s2[1]],
#                        [u_refl_p[2], u_refl_s1[2], u_refl_s2[2],
#                         - u_trans_p[2], - u_trans_s1[2], - u_trans_s2[2]],
#                        [rp_coeff[0], rs1_coeff[0], rs2_coeff[0],
#                         - tp_coeff[0], - ts1_coeff[0], - ts2_coeff[0]],
#                        [rp_coeff[1], rs1_coeff[1], rs2_coeff[1],
#                         - tp_coeff[1], - ts1_coeff[1], - ts2_coeff[1]],
#                        [rp_coeff[2], rs1_coeff[2], rs2_coeff[2],
#                         - tp_coeff[2], - ts1_coeff[2], - ts2_coeff[2]]])
#
#     # Введём правую часть системы уравнений:
#
#     right_part = np.array([- u0[0],
#                            - u0[1],
#                            - u0[2],
#                            - u0_coeff[0],
#                            - u0_coeff[1],
#                            - u0_coeff[2]])
#
#     # Решим эту систему уравнений:
#     rp, rs1, rs2, tp, ts1, ts2 = np.linalg.solve(matrix, right_part)
#
#     # И вернём результат:
#
#     if rt_signum == 1:
#
#         return tp, ts1, ts2
#
#     else:
#
#         return rp, rs1, rs2

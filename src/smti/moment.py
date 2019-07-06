import numpy as np


def _shear_moment(strike, dip, rake, ds, n):
    """
    M11 = −DS N (sin2ϕsinδcosλ + sin^2ϕsin2δsinλ)
    M12 = +DS N (cos2ϕsinδcosλ +0.5sin2ϕsin2δsinλ)
    M13 = −DS N (cosϕcosδcosλ + sinϕcos2δsinλ)
    M22 = +DS N (sin2ϕsinδcosλ − cos2ϕsin2δsinλ)
    M23 = −DS N (sinϕcosδcosλ − cosϕcos2δsinλ)
    M33 = +DS N sin2δsinλ

    :param strike:  ϕ
    :param dip: δ
    :param rake: λ
    :param ds: the magnitude of the shear (DS) dislocation
    :return: np.array([m11, m12, m13, m22, m23, m33])
    """
    from numpy import sin, cos

    m11 = - (sin(2 * strike) * sin(dip) * cos(rake) + (sin(strike) ** 2) * sin(2 * dip) * sin(rake))
    m12 = + (cos(2 * strike) * sin(dip) * cos(rake) + 0.5 * sin(2 * strike) * sin(2 * dip) * sin(rake))
    m13 = - (cos(strike) * cos(dip) * cos(rake) + sin(strike) * cos(2 * dip) * sin(rake))
    m22 = + (sin(2 * strike) * sin(dip) * cos(rake) - (cos(strike) ** 2) * sin(2 * dip) * sin(rake))
    m23 = - (sin(strike) * cos(dip) * cos(rake) - cos(strike) * cos(2 * dip) * sin(rake))
    m33 = + (sin(2 * dip) * sin(rake))

    return ds * n * np.array([m11, m12, m13, m22, m23, m33])


def _opening_moment(strike, dip, dn, n, ng):
    """
    M11 = DN (η + 2Nsin^2ϕsin^2δ)
    M12 = −DN Nsin2ϕsin^2δ
    M13 = DN Nsinϕsin2δ
    M22 = DN (η + 2Ncos2ϕsin2δ)
    M23 = −DN Ncosϕsin2δ
    M33 = +DN (η + 2Ncos^2δ)

    :param strike:
    :param dip:
    :param rake:
    :param n:
    :param ng:
    :return:
    """

    from numpy import sin, cos

    m11 = + dn * (ng + 2 * n * (sin(strike) ** 2) * (sin(dip) ** 2))
    m12 = - dn * n * sin(2 * strike) * (sin(dip) ** 2)
    m13 = + dn * n * sin(strike) * sin(2 * dip)
    m22 = + dn * (ng + 2 * n * (np.cos(strike) ** 2) * (np.sin(dip) ** 2))
    m23 = - dn * n * cos(strike) * sin(2 * dip)
    m33 = + dn * (ng + 2 * n * (cos(dip) ** 2))

    return np.array([m11, m12, m13, m22, m23, m33])


def create_general_moment(strike, dip, rake, ds=1, dn=0, n=1, ng=1.4, a=1):
    """
    M11 = −DS N (sin2ϕsinδcosλ + sin^2ϕsin2δsinλ) +DN (η + 2Nsin^2ϕsin^2δ)
    M12 = +DS N (cos2ϕsinδcosλ +0.5sin2ϕsin2δsinλ) −DN Nsin2ϕsin^2δ
    M13 = −DS N (cosϕcosδcosλ + sinϕcos2δsinλ) +DN Nsinϕsin2δ
    M22 = +DS N (sin2ϕsinδcosλ − cos2ϕsin2δsinλ) +DN (η + 2Ncos2ϕsin2δ)
    M23 = −DS N (sinϕcosδcosλ − cosϕcos2δsinλ) −DN Ncosϕsin2δ
    M33 = +DS N sin2δsinλ +DN (η + 2Ncos2δ)

    output: m = np.array([m11, m12, m13, m22, m23, m33])

    :param strike: strike (ϕ), rad
    :param dip: dip (δ), rad,
    :param rake: rake/slip (λ), rad
    :param ds: the magnitude of the shear (DS) dislocation, m
    :param dn: the magnitude of the opening (DN) dislocation, m
    :param n:  1st Lame's Constant, Pa
    :param ng: 2nd Lame's Constant, Pa
    :param a: surface of source, m^2
    :return:
    """
    if np.abs(a) < .0001:
        raise ValueError('Surface of source "a" must be bigger')

    m = _shear_moment(strike, dip, rake, ds, n) + _opening_moment(strike, dip, dn, n, ng)
    m = m / np.abs(a)
    return m


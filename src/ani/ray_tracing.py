import numpy as np


def reflected_time_iso(ray, sou, rec, thk, vel, n):
    sou = np.array(sou)
    rec = np.array(rec)
    ray = np.array(ray)

    idx = np.arange(n).ravel()
    idx = np.hstack([idx, idx[::-1]])
    x = np.hstack([sou.ravel(), ray.ravel(), rec.ravel()])

    x = x[1:] - x[:-1]
    y = thk[idx]

    r = np.sqrt(x ** 2 + y ** 2)

    return np.sum(r / vel[idx])


def head_wave_iso(vel, thk, offset):
    m = len(vel)

    t = np.zeros((m - 1, len(offset)))
    xnm = np.zeros((m - 1, 1))
    tnm = np.zeros((m - 1, 1))
    t0m = np.zeros((m - 1, 1))
    for i in range(m - 1):
        al = np.arcsin(vel[:(i + 1)] / vel[i + 1])
        _xnm = 2 * np.sum(thk[:(i + 1)] * np.tan(al))
        _tnm = 2 * np.sum(thk[:(i + 1)] / np.cos(al) / vel[:(i + 1)])

        _t0m = 2 * np.sum(thk[:(i + 1)] * np.cos(al) / vel[:(i + 1)])
        t[i] = _t0m + offset / vel[i + 1]
        t[i, offset < _xnm] = np.nan

        xnm[i] = _xnm
        tnm[i] = _tnm
        t0m[i] = _t0m

    return np.row_stack([offset / vel[0], t]), xnm, tnm, t0m

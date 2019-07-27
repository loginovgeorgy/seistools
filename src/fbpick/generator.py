import numpy as np


def _berlage(t, t0, f, g):
    w = np.sin(2 * f * np.pi * (t - t0))
    h = np.exp(-g * (t - t0) ** 2)

    s = w * h
    s[t < t0] = 0
    s = s / np.max(np.abs(s))
    return s


def generate_data_set(nr, ns,
                      dt=1.,
                      t0_min=.25,
                      t0_max=.8,
                      g_min=2,
                      g_max=4,
                      f_max=.3,
                      f_min=.1,
                      n_min=.01,
                      n_max=.2):
    fn = 1. / 2. / dt
    n = np.random.uniform(n_min, n_max, nr)[np.newaxis].T

    t0 = dt * np.random.uniform((ns - 1) * t0_min, (ns - 1) * t0_max, nr)[np.newaxis].T
    f = np.random.uniform(fn * f_min, fn * f_max, nr)[np.newaxis].T
    al = np.random.randint(g_min, g_max + 1, nr)[np.newaxis].T
    g = f / al

    noise = np.random.randn(nr, ns) * n
    t = np.arange(0, ns * dt, dt)
    t = np.repeat(t[np.newaxis], nr, axis=0)
    x = _berlage(t, t0, f, g)
    x_noise = x + noise
    y = t0
    x = np.sign(np.random.uniform(-1, 1, nr))[np.newaxis].T * x

    return x, x_noise, y


def _augm(x,y, na):
    nr, ns = x.shape
    jj = np.random.randint(0,nr-1,na)
    xx, yy = [], []
    for k in jj:
        _x = x[k]
        _y = y[k]
        j = _y.argmax()
        s = min([100, j-5])
        t = np.roll(_x, s)
        t[0:s] = 1e-4*np.random.rand(1)
        xx.append(t)
        yy.append(np.roll(_y, s))
    xx = np.array(xx)
    yy = np.array(yy)
    return xx, yy

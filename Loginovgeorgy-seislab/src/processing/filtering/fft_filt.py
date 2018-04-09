import numpy as np


def forward_fourier(x, axis=1):
    x = np.fft.ifftshift(x, axes=axis)
    x = np.fft.fft(x, axis=axis)
    x = np.fft.fftshift(x, axes=axis)
    return x


def inverse_fourier(x, axis=1):
    x = np.fft.ifftshift(x, axes=axis)
    x = np.fft.ifft(x, axis=axis)
    x = np.fft.ifftshift(x, axes=axis)
    return np.real(x)


def band_pass(x, b, dt, axis=1):
    ns = x.shape[axis]

    ft = build_filter(b, ns, dt)
    z = forward_fourier(x, axis)
    x = inverse_fourier(z * ft, axis)
    return x


def build_filter(b, ns, dt):
    b = np.array(b)

    do = 1 / ((ns - 1) * dt)
    nw = np.int32(np.floor(ns / 2) + 1)

    bj = np.int32(np.floor(b / do) + 1)
    bj[bj > nw] = nw
    bj[bj < 0] = 0

    n1 = bj[1] - bj[0]
    n2 = bj[3] - bj[2]

    ft = np.zeros((nw, 1))

    ft[bj[0]:bj[1]:1, 0] = np.linspace(0, 1, n1)
    ft[bj[2]:bj[3]:1, 0] = np.linspace(1, 0, n2)
    ft[bj[1]:bj[2]] = 1

    if np.mod(ns, 2) == 0:
        ftr = ft[1:-1]
    else:
        ftr = ft[1:]
    ft = np.vstack((ft[-1::-1], ftr))
    return ft

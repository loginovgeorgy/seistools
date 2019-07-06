import matplotlib.pyplot as plt


def plot_moments(x, centers=None, labels=None, s=40):
    c_clvd, c_iso = x[:, 0], x[:, 1]

    if labels is not None:
        if not isinstance(labels, list):
            labels = [labels]

    if centers is not None:
        if not isinstance(centers, list):
            centers = [centers]

    plt.figure(figsize=(15, 5))

    for i, (l, cn) in enumerate(zip(labels, centers)):
        axes = plt.subplot(1, len(labels), i + 1)
        plt.plot([1, 0, -1, 0, 1],
                 [0, 1, 0, -1, 0], 'k')

        plt.scatter(c_clvd, c_iso, c=l, s=s)
        plt.plot(cn[:, 0],
                 cn[:, 1], '*r', ms=10)

        axes.set_xlim([-1.1, 1.1])
        axes.set_ylim([-1.1, 1.1])
        axes.set_aspect('equal')


def plot_moments_vu(x, centers=None, labels=None, s=40):
    c_clvd, c_iso = x[:, 0], x[:, 1]

    if labels is not None:
        if not isinstance(labels, list):
            labels = [labels]

    if centers is not None:
        if not isinstance(centers, list):
            centers = [centers]

    plt.figure(figsize=(15, 5))

    for i, temp in enumerate(zip(labels, centers)):
        l, cn = temp
        axes = plt.subplot(1, len(labels), i + 1)
        plt.plot([-4 / 3, 0, 4 / 3, 0, -4 / 3],
                 [-1 / 3, 1, 1 / 3, -1, -1 / 3], 'k')

        plt.scatter(c_clvd, c_iso, c=l, s=s)
        plt.plot(cn[:, 0],
                 cn[:, 1], '*r', ms=10)

        axes.set_xlim([-1.1 * 4 / 3, 1.1 * 4 / 3])
        axes.set_ylim([-1.1, 1.1])
        axes.set_aspect('equal')
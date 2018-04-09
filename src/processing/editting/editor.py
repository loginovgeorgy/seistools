def normalize_data(x, axis=2):
    x = x.copy()
    shape = x.shape
    if len(shape) == 2:
        mu = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        x = (x - mu) / std

    elif len(shape) == 3:
        if shape[-1] == 1:
            mu = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            x = (x - mu) / std

        elif shape[-1] > 1:
            mu = x.mean(axis=axis, keepdims=True)
            std = x.std(axis=axis, keepdims=True)
            x = (x - mu) / std

    return x

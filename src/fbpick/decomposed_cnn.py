import numpy as np


def _convolution(signal, weight):
    signal = signal.copy()
    weight = weight.copy()

    signal = signal.ravel()
    weight = weight.ravel()

    ns = len(signal)
    nf = len(weight) / 2
    nf = np.floor(nf)
    nf = np.int32(nf)

    result = np.correlate(signal, weight, mode='full')
    result = result[nf:]
    result = result[:ns]
    return result


def _apply_filters(signal, weight):
    assert len(signal.shape) == 2, (
        'Input must by 2D array. ',
        'The given input shape {}'.format(signal.shape)
    )

    nw = weight.shape[2]
    ns = signal.shape[0]
    nc = signal.shape[1]

    result = np.zeros((ns, nw))

    for ic in range(nc):
        for iw in range(nw):
            result[:, iw] += _convolution(signal[:, ic], weight[:, ic, iw])

    return result


def _conv1d_layer(layer, signal):
    signal = signal.copy()

    layer_weights = layer['weights']

    filters = layer_weights[0]

    signal = _fftconvolution(signal, filters)

    if layer['use_bias']:
        bias = layer_weights[1]
        if len(signal.shape) != len(bias.shape):
            bias = bias[None, ...]

        signal += bias

    activation = layer['activation']
    activation_function = ACTIVATIONS[activation]

    return activation_function(signal)


def _fftconvolution(signal, weight, axis=1):
    from scipy.signal import fftconvolve
    signal = signal.copy()
    weight = weight.copy()

    if len(signal.shape) != len(weight.shape):
        raise ValueError(
            'signal.shape = {}, weight.shape = {}'.format(
                signal.shape,
                weight.shape,
            )
        )

    nr, ns, nc = signal.shape
    nw, nc, nf = weight.shape
    nw2 = np.floor(nw / 2)
    nw2 = np.int32(nw2)

    signal = np.flip(signal, axis=axis)

    signal = signal[..., None]
    weight = weight[None, ...]

    result = fftconvolve(signal, weight, mode='full', axes=(axis))
    result = np.flip(result, axis=axis)
    result = result.sum(axis=2)

    result = result[:, nw2:]
    result = result[:, :ns]

    return result


def _batchnorm_layer(layer, signal, eps=1e-16):
    signal = signal.copy()
    layer_weights = layer['weights']
    weights = np.array(layer_weights)
    mu, sigma = weights[-2:]

    if len(signal.shape) != len(mu.shape):
        mu = mu[None, ...]
        sigma = sigma[None, ...]

    signal -= mu
    signal /= np.sqrt(sigma + eps)

    if layer['scale']:
        signal *= weights[0]

    if layer['center']:
        signal += weights[1]

    return signal


def _activation_layer(layer, signal):
    activation = layer['activation']
    activation_function = ACTIVATIONS[activation]
    return activation_function(signal)


ACTIVATIONS = {
    'relu': lambda x: np.maximum(0, x),
    'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-x)),
    'linear': lambda x: x,
    'softmax': lambda x: np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True),
    'tanh': lambda x: np.tanh(x),
}

LAYERS = {
    'conv1d': _conv1d_layer,
    'batchnorm': _batchnorm_layer,
    'activation': _activation_layer,
}

LAYERS_NEW = {
    'conv1d': _conv1d_layer,
    'batchnorm': _batchnorm_layer,
    'activation': _activation_layer,
}


def apply_cnn_model(model, signal):
    signal = signal.copy()
    signal = np.float32(signal)

    for i, layer in enumerate(model['layers']):
        layer_name = layer['name']
        func = LAYERS_NEW[layer_name]
        signal = func(layer, signal)
    return signal


def summary_conv1d(l):
    summary = 'CONV1D. f: {}, w: {}, use_bias: {}'.format(
        l['activation'],
        l['weights'][0].shape,
        l['use_bias'],
    )
    return summary


def summary_activation(l):
    summary = 'ACTIVATION. f: {}'.format(
        l['activation'],
    )
    return summary


def summary_batchnorm(l):
    summary = 'BATCHNORM. w: {}'.format(
        np.array(l['weights']).shape,
    )
    return summary


SUMMARY = {
    'conv1d': summary_conv1d,
    'activation': summary_activation,
    'batchnorm': summary_batchnorm,
}


def model_summary(model_dict):
    print('>>> input_shape:', model_dict['input_shape'])
    print('>>> output_shape:', model_dict['output_shape'])
    for i, l in enumerate(model_dict['layers']):
        print(i, SUMMARY[l['name']](l))

import keras
from keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from  itertools import product


def weighted_categorical_crossentropy2(weights):
    def loss(y_true, y_pred):
        nb_cl = len(weights)
        final_mask = keras.backend.zeros_like(y_pred[:, 0])
        y_pred_max = keras.backend.max(y_pred, axis=1, keepdims=True)
        y_pred_max_mat = keras.backend.equal(y_pred, y_pred_max)
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return keras.backend.categorical_crossentropy(y_pred, y_true) * final_mask

    return loss


def weighted_binary_crossentropy(zero_weight):
    # def f(y_true, y_pred):
    #     return mean(binary_crossentropy(y_true, y_pred), axis=-1)
    # return f
    one_weight = 1 - zero_weight

    def loss(y_true, y_pred):
        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_tensor = y_true[..., 0] * one_weight + (1. - y_true[..., 0]) * zero_weight
        weighted_b_ce = weight_tensor * b_ce

        # Return the mean error
        return keras.backend.mean(weighted_b_ce)

    return loss


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = keras.backend.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        # calc
        loss = y_true * keras.backend.log(y_pred) * weights
        loss = -keras.backend.sum(loss, -1)
        return loss

    return loss


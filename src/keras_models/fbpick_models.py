from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D

from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, GlobalAveragePooling1D, GlobalMaxPooling1D,
from keras.layers import AveragePooling1D

from keras.layers import Activation

from keras.layers import Input, Dense, Flatten, Reshape, Permute, BatchNormalization, Dropout, Concatenate, merge
from keras.optimizers import RMSprop
from keras.models import model_from_json, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Lambda
from keras.losses import binary_crossentropy, sparse_categorical_crossentropy
import numpy as np
from .losses import *
import keras
import tensorflow


POOLING = {
    'max': MaxPooling1D,
    'average': AveragePooling1D,
}


def conv1d_block(
        layer,
        filters,
        kernel_size,
        activation,
        dropout=None,
        batch_norm=True,
        pooling=None,
        upsampling=None,
        pooling_type='max',
        seed=37
):

    x = Conv1D(
        filters,
        kernel_size=kernel_size,
        activation=None,
        padding='same'
    )(layer)
    x = Activation(activation)
    x = BatchNormalization()(x) if batch_norm else x
    x = Dropout(dropout, seed=seed)(x) if dropout else x
    x = POOLING[pooling_type](pooling) if pooling else x
    x = UpSampling1D(upsampling) if upsampling else x

    return x


def model_conv1d(
        n_cl,
        shape=None,
        depth=6,
        filters=32,
        kernel_size=32,
        channels=1,
        activation='relu',
        last_activation='softmax',
        last_kernel_size=32,
        dropout=.1,
        batch_norm=True,
        pooling=None,
        lr=.001
):
    input_img = Input(shape=(shape, channels))

    x = input_img
    for i in range(depth):
        x = conv1d_block(
            x,
            filters,
            kernel_size,
            activation,
            dropout=dropout,
            batch_norm=batch_norm,
            pooling=pooling,
            upsampling=None,
            pooling_type='max',
            seed=37,
        )

    x = conv1d_block(
        x,
        n_cl,
        last_kernel_size,
        last_activation,
        dropout=dropout,
        batch_norm=batch_norm,
        pooling=None,
        upsampling=None,
        pooling_type='max',
        seed=37,
    )

    model = Model(input_img, x, name="conv_segm")

    optimizer = keras.optimizers.Adam(
        lr=lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0,
        amsgrad=False
    )

    loss = 'categorical_crossentropy'
    # loss = custom_loss
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model

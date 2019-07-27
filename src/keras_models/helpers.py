import keras
from IPython.display import clear_output
import pylab as plt
from keras.models import model_from_json
from itertools import product
import tensorflow


class PlotLosses(keras.callbacks.Callback):
    def __init__(self, title=''):
        self.title = title

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        if self.i < self.params['epochs']:
            clear_output(wait=True)

            plt.plot(self.x, self.losses, label="training")
            plt.plot(self.x, self.val_losses, label="validation")
            plt.xlabel('No of epochs')
            plt.ylabel('Loss')
            plt.title(self.title)
            plt.legend()
            plt.show()


def load_keras_model(name):
    path = "{}.json".format(name)
    with open(path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json, custom_objects={"tensorflow": tensorflow})
    # load weights into new model
    path = "{}.h5".format(name)
    model.load_weights(path)
    return model


def save_keras_model(model, name):
    path = "{}.json".format(name)
    model_json = model.to_json()
    with open(path, "w") as json_file:
        json_file.write(model_json)

    path = "{}.h5".format(name)
    model.save_weights(path)
    return


def _parse_conv1d(layer):
    return {
        'name': 'conv1d',
        'weights': layer.get_weights(),
        'activation': layer.get_config()['activation'],
        'use_bias': layer.use_bias,
    }


def _parse_batchnorm(layer):
    return {
        'name': 'batchnorm',
        'weights': layer.get_weights(),
        'scale': layer.scale,
        'center': layer.center,
    }


def _parse_activation(layer):
    return {
        'name': 'activation',
        'activation': layer.get_config()['activation'],
    }


PARSING = {
    'conv1d': _parse_conv1d,
    'batch': _parse_batchnorm,
    'activation': _parse_activation,
}


def convert_model_to_dict(model):
    layers = []
    for i, l in enumerate(model.layers):
        name = l.name.split('_')[0]
        if name not in list(PARSING.keys()):
            continue

        layers += [PARSING[name](l)]

    model_dict = {
        'output_shape': model.output_shape,
        'input_shape': model.input_shape,
        'layers': layers,
    }
    return model_dict

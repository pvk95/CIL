from tensorflow import keras
import numpy as np
import pandas as pd


class UNet(object):
    def __init__(self, input_shape=(400, 400, 3), epochs=30, verbose=1, batch_size=4, deepness=4):
        self.input_shape = input_shape
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.deepness = deepness

    def create_model(self):
        self.input = keras.layers.Input(self.input_shape)
        self.skip = []
        inp = self.input
        # Convolution
        for x in range(self.deepness):
            filters = 2**(6 + x)
            skip, inp = self.conv_layer(filters, inp)
            self.skip.append(skip)

        # lowest layer
        conv1 = keras.layers.Conv2D(
            self.deepness, 3, activation='relu', padding='same')(inp)
        conv2 = keras.layers.Conv2D(
            self.deepness, 3, activation='relu', padding='same')(conv1)

        # Upsample and convolutions
        inp = conv2
        for x in range(self.deepness - 1, -1, -1):
            filters = 2**(6 + x)
            inp = self.upconv_layer(filters, inp, self.skip[x])

        output = keras.layers.Conv2D(1, 1, activation='sigmoid')(inp)
        model = keras.models.Model(inputs=self.input, outputs=output)
        model.summary()
        return model

    def conv_layer(self, filters, inp):
        conv1 = keras.layers.Conv2D(
            filters, 3, activation='relu', padding='same')(inp)
        conv2 = keras.layers.Conv2D(
            filters, 3, activation='relu', padding='same')(conv1)
        max_pool = keras.layers.MaxPool2D(2, strides=2)(conv2)
        return conv2, max_pool

    def upconv_layer(self, filters, inp, skip):
        up_conv = keras.layers.Conv2DTranspose(filters, 2, 2)(inp)
        up_shape = up_conv.shape.as_list()
        skip_shape = skip.shape.as_list()

        x_start = (skip_shape[1] - up_shape[1]) // 2
        y_start = (skip_shape[2] - up_shape[2]) // 2
        x_end = x_start + up_shape[1]
        y_end = y_start + up_shape[2]

        cut_skip = keras.layers.Lambda(
            lambda x: x[:, x_start:x_end, y_start: y_end, :])(skip)

        merge = keras.layers.concatenate([cut_skip, up_conv], axis=-1)
        conv1 = keras.layers.Conv2D(
            filters, 3, activation='relu', padding='same')(merge)
        conv2 = keras.layers.Conv2D(
            filters, 3, activation='relu', padding='same')(conv1)

        return conv2

    def fit(self, X, y):
        self.model = self.create_model()
        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(x=X, y=y, batch_size=self.batch_size, verbose=self.verbose,
                       epochs=self.epochs)
        return self

    def predict(self, X):
        return self.model.predict(X, batch_size=self.batch_size)

    def get_params(self, deep=True):
        return {
            'input_shape': self.input_shape,
            'epochs': self.epochs,
            'verbose': self.verbose,
            'batch_size': self.batch_size,
            'deepness': self.deepness
        }

    def set_params(self, **paramters):
        for paramter, value in paramters.items():
            setattr(self, paramter, value)

    def train(self, X_train, Y_train, X_valid, Y_valid):
        self.model = self.create_model()
        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
                       batch_size=self.batch_size, verbose=self.verbose, epochs=self.epochs)
        return self

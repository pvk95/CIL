import datetime
import os
import pickle
import sys

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.models import Model


class SegNet():
    def __init__(self, im_sz=32, n_channels=3, lr=0.001, n_epochs=100, save_folder='SegNet/', batch_sz=32):
        self.im_sz = im_sz
        self.n_channels = n_channels
        self.lr = lr
        self.n_epochs = n_epochs
        self.save_folder = save_folder
        self.batch_sz = batch_sz
        self.main_input = Input(shape=[im_sz, im_sz, n_channels])

        # Create the model
        self.model = self.create()

    def add_conv_layer(self, input, n_filters, flt_sz=3, stride=1, n_conv=2):

        xs = [input]
        for i in range(n_conv):
            output = Conv2D(filters=n_filters, kernel_size=(flt_sz, flt_sz), strides=stride, \
                            padding="same")(xs[-1])

            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            xs.append(output)

        output = MaxPool2D()(xs[-1])

        return output

    def add_deconv_layer(self, input, n_filters, flt_sz=3, strides=2, n_conv=3, last=False):

        xs = [input]
        output = Conv2DTranspose(filters=n_filters, kernel_size=flt_sz, \
                                 strides=strides, padding='same')(input)
        xs.append(output)

        for i in range(n_conv):
            output = Conv2D(filters=n_filters, kernel_size=(flt_sz, flt_sz), strides=1, \
                            padding="same")(xs[-1])
            output = BatchNormalization()(output)

            if (last and i == n_conv - 1):
                output = Activation('softmax')(output)
            else:
                output = Activation('relu')(output)

            xs.append(output)

        return xs[-1]

    def create(self):
        layer_1 = self.add_conv_layer(self.main_input, 20)
        layer_2 = self.add_conv_layer(layer_1, 40)
        layer_3 = self.add_conv_layer(layer_2, 80)

        deconv_1 = self.add_deconv_layer(layer_3, 40)
        deconv_2 = self.add_deconv_layer(deconv_1, 20)
        self.main_output = self.add_deconv_layer(deconv_2, 1, last=True)

        model = Model(self.main_input, self.main_output)

        opt = tf.keras.optimizers.Adam(lr=self.lr)
        binary_loss = tf.keras.losses.binary_crossentropy
        model.compile(optimizer=opt, loss=binary_loss)

        return model

    def train(self, X_train, y_train, X_valid, y_valid):
        model = self.model
        history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                            epochs=self.n_epochs, batch_size=self.batch_sz)

        training_loss = history.history['loss']
        val_loss = history.history['val_loss']

        train_curves = {'train': training_loss, 'val': val_loss}

        with open(self.save_folder + 'train_curves.pickle', 'wb') as f:
            pickle.dump(train_curves, f)

        if not os.path.exists(self.save_folder + 'checkpoint'):
            os.makedirs(self.save_folder + 'checkpoint')

        fileName = self.save_folder + 'checkpoint/SegNet.h5'

        tf.keras.models.save_model(model, filepath=fileName)

    def predict(self, X_test):
        fileName = self.save_folder + 'checkpoint/SegNet.h5'
        if not os.path.isfile(fileName):
            print("Model not found! Exiting ...")
            sys.exit(1)

        model = tf.keras.models.load_model(fileName)
        y_pred = model.predict(X_test)
        y_pred = (y_pred >= 0.5).astype(np.int)

        with h5py.File(self.save_folder + 'predictions.h5', 'w') as f:
            f['data'] = y_pred

        return y_pred


class ResUNet():
    def __init__(self, save_folder, input_shape=(400, 400, 3), epochs=30, verbose=1, batch_size=4, deepness=4):
        self.input_shape = input_shape
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.deepness = deepness
        self.save_folder = save_folder
        self.model_name = "res-unet"
        self.model_file = self.model_name + datetime.datetime.today().strftime("_%d_%m_%y_%H:%M:%S") + ".h5"

    def res_block(self, x, nb_filters, strides):
        res_path = keras.layers.BatchNormalization()(x)
        res_path = keras.layers.Activation(activation='relu')(res_path)
        res_path = keras.layers.Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(
            res_path)
        res_path = keras.layers.BatchNormalization()(res_path)
        res_path = keras.layers.Activation(activation='relu')(res_path)
        res_path = keras.layers.Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(
            res_path)

        shortcut = keras.layers.Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
        shortcut = keras.layers.BatchNormalization()(shortcut)

        res_path = keras.layers.add([shortcut, res_path])
        return res_path

    def encoder(self, x):
        to_decoder = []

        main_path = keras.layers.Conv2D(filters=50, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
        main_path = keras.layers.BatchNormalization()(main_path)
        main_path = keras.layers.Activation(activation='relu')(main_path)

        main_path = keras.layers.Conv2D(filters=50, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

        shortcut = keras.layers.Conv2D(filters=50, kernel_size=(1, 1), strides=(1, 1))(x)
        shortcut = keras.layers.BatchNormalization()(shortcut)

        main_path = keras.layers.add([shortcut, main_path])
        # first branching to decoder
        to_decoder.append(main_path)

        main_path = self.res_block(main_path, [100, 100], [(2, 2), (1, 1)])
        to_decoder.append(main_path)

        main_path = self.res_block(main_path, [200, 200], [(2, 2), (1, 1)])
        to_decoder.append(main_path)

        return to_decoder

    def decoder(self, x, from_encoder):
        main_path = keras.layers.UpSampling2D(size=(2, 2))(x)
        main_path = keras.layers.concatenate([main_path, from_encoder[2]], axis=3)
        main_path = self.res_block(main_path, [200, 200], [(1, 1), (1, 1)])

        main_path = keras.layers.UpSampling2D(size=(2, 2))(main_path)
        main_path = keras.layers.concatenate([main_path, from_encoder[1]], axis=3)
        main_path = self.res_block(main_path, [100, 100], [(1, 1), (1, 1)])

        main_path = keras.layers.UpSampling2D(size=(2, 2))(main_path)
        main_path = keras.layers.concatenate([main_path, from_encoder[0]], axis=3)
        main_path = self.res_block(main_path, [50, 50], [(1, 1), (1, 1)])

        return main_path

    def create_model(self):
        self.input = keras.layers.Input(shape=self.input_shape)

        to_decoder = self.encoder(x=self.input)

        path = self.res_block(x=to_decoder[2], nb_filters=[400, 400], strides=[(2, 2), (1, 1)])
        path = self.decoder(x=path, from_encoder=to_decoder)
        path = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path)

        model = keras.models.Model(inputs=self.input, outputs=path)
        model.summary()
        return model

    def predict(self, X):
        file_name = self.save_folder + 'checkpoint/UNet.h5'
        if not os.path.isfile(file_name):
            print("Model not found! Exiting ...")
            sys.exit(1)
        self.model = tf.keras.models.load_model(file_name)
        y_pred = self.model.predict(X, batch_size=self.batch_size)
        y_pred = (y_pred >= 0.5).astype(np.int)

        with h5py.File(self.save_folder + 'predictions.h5', 'w') as f:
            f['data'] = y_pred

    def get_params(self):
        return {
            'model_name': self.model_name,
            'input_shape': self.input_shape,
            'epochs': self.epochs,
            'verbose': self.verbose,
            'batch_size': self.batch_size,
            'deepness': self.deepness
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

    def train(self, X_train, Y_train, X_valid, Y_valid):
        self.model = self.create_model()
        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        model_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(self.save_folder, self.model_file),
                                                           monitor='loss', save_best_only=True, verbose=True)
        tensor_board = keras.callbacks.TensorBoard()
        history = self.model.fit(x=X_train, y=Y_train, batch_size=self.batch_size, verbose=self.verbose,
                                 epochs=self.epochs,
                                 callbacks=[tensor_board, model_checkpoint], validation_data=(X_valid, Y_valid))

        training_loss = history.history['loss']
        val_loss = history.history['val_loss']

        train_curves = {'train': training_loss, 'val': val_loss}

        with open(self.save_folder + 'train_curves.pickle', 'wb') as f:
            pickle.dump(train_curves, f)

        if not os.path.exists(self.save_folder + 'checkpoint/'):
            os.makedirs(self.save_folder + 'checkpoint')

        file_name = self.save_folder + 'checkpoint/Res_UNet.h5'
        tf.keras.models.save_model(self.model, filepath=file_name)

        return self

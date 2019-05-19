import tensorflow as tf
import os
import sys
import h5py
from tensorflow import keras
import numpy as np
import pickle
import datetime

# copy and paste from: https://github.com/DuFanXin/deep_residual_unet/blob/master/res_unet.py


class ResUNet(object):
    def __init__(self, save_folder, input_shape=(400, 400, 3), epochs=30, verbose=1, batch_size=4, deepness=4):
        self.input_shape = input_shape
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.deepness = deepness
        self.save_folder = save_folder
        self.model_name = "res-unet_"
        self.model_file = self.model_name + datetime.datetime.today().strftime("_%d_%m_%y_%H:%M:%S") + ".h5"

    def res_block(self, x, nb_filters, strides):
        res_path = keras.layers.BatchNormalization()(x)
        res_path = keras.layers.Activation(activation='relu')(res_path)
        res_path = keras.layers.Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
        res_path = keras.layers.BatchNormalization()(res_path)
        res_path = keras.layers.Activation(activation='relu')(res_path)
        res_path = keras.layers.Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

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

    def fit(self, X, y):
        self.model = self.create_model()
        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        model_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(self.save_folder, self.model_file),
                                                           monitor='loss', save_best_only=True, verbose=True)
        tensor_board = keras.callbacks.TensorBoard()

        self.model.fit(x=X, y=y, batch_size=self.batch_size, verbose=self.verbose, epochs=self.epochs,
                       callbacks=[tensor_board, model_checkpoint])
        return self

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
        history = self.model.fit(x=X_train, y=Y_train, batch_size=self.batch_size, verbose=self.verbose, epochs=self.epochs,
                                 callbacks=[tensor_board, model_checkpoint], validation_data=(X_valid, Y_valid))

        # history = self.model.fit(x=X_train, y=Y_train, batch_size=self.batch_size, verbose=self.verbose, epochs=self.epochs,
        #                          validation_data=(X_valid, Y_valid))

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


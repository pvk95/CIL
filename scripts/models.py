import os
import sys
import pickle
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Activation, Dropout, Conv2DTranspose, Input, Conv2D,
    BatchNormalization, MaxPool2D, Concatenate, Flatten,
    Reshape, Dense, SpatialDropout2D)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras.backend import set_session
from tensorflow.keras.models import Model, load_model


class getModel(object):
    def __init__(self, save_folder='./', epochs=30, verbose=1, batch_size=4, model_name='UNet.h5'):

        # Training specific params
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.save_folder = save_folder
        self.model_name = model_name

    def train(self, X_train, Y_train, X_valid, Y_valid):

        #config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        #sess = tf.Session(config=config)
        # set this TensorFlow session as the default session for Keras
        # set_session(sess)

        early = EarlyStopping(monitor="val_acc", mode="max",
                              patience=10, verbose=self.verbose)
        redonplat = ReduceLROnPlateau(
            monitor="val_acc", mode="max", patience=5, verbose=self.verbose)

        history = self.model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
                                 batch_size=self.batch_size, verbose=self.verbose, epochs=self.epochs)

        training_loss = history.history['loss']
        val_loss = history.history['val_loss']

        train_curves = {'train': training_loss, 'val': val_loss}

        with open(self.save_folder + 'train_curves.pickle', 'wb') as f:
            pickle.dump(train_curves, f)

        if not os.path.exists(self.save_folder + 'checkpoint/'):
            os.makedirs(self.save_folder + 'checkpoint')

        fileName = self.save_folder + 'checkpoint/' + self.model_name
        tf.keras.models.save_model(self.model, filepath=fileName)

    def predict(self, X):
        fileName = self.save_folder + 'checkpoint/' + self.model_name
        if not os.path.isfile(fileName):
            print("Model not found! Exiting ...")
            sys.exit(1)
        self.model = tf.keras.models.load_model(fileName)
        y_pred = self.model.predict(X, batch_size=self.batch_size)
        y_pred = (y_pred >= 0.5).astype(np.int)

        return y_pred


class SegNet(getModel):
    def __init__(self, save_folder='./', lr=0.001, input_shape=(400, 400, 3), epochs=30, verbose=1,
                 batch_size=32, model_name='SegNet.h5'):

        # Model specific params
        self.lr = lr
        self.input_shape = input_shape
        getModel.__init__(self, save_folder, epochs,
                          verbose, batch_size, model_name)

        # Create and compile the model SegNet
        self.model = self.create_model()

    def add_conv_layer(self, input, n_filters, flt_sz=5, stride=1, n_conv=2, num=0):

        xs = [input]
        for i in range(n_conv):
            dropout = SpatialDropout2D(0.1, name=f'dropout_{i}_{num}_segnet')(xs[-1])
            output = Conv2D(filters=n_filters, kernel_size=(flt_sz, flt_sz), strides=stride,
                            padding="same", name=f'conv2d_{i}_{num}_segnet')(dropout)

            output = BatchNormalization(name=f'bn_{i}_{num}_segnet')(output)
            output = Activation('relu', name=f'relu_{i}_{num}_segnet')(output)
            xs.append(output)

        output = MaxPool2D(name=f'maxpool_{num}')(xs[-1])

        return output

    def add_deconv_layer(self, input, n_filters, flt_sz=5, strides=2, n_conv=3, last=False, num=0):

        xs = [input]
        output = Conv2DTranspose(filters=n_filters, kernel_size=flt_sz,
                                 strides=strides, padding='same', name=f'up_conv2tranpose_{num}_segnet')(input)
        xs.append(output)

        for i in range(n_conv):
            dropout = SpatialDropout2D(0.1, name=f'up_dropout_{i}_{num}_segnet')(xs[-1])
            output = Conv2D(filters=n_filters, kernel_size=(flt_sz, flt_sz), strides=1,
                            padding="same", name=f'up_conv2d_{i}_{num}_segnet')(dropout)
            output = BatchNormalization(name=f'up_bn_{i}_{num}_segnet')(output)

            if (last and i == n_conv - 1):
                output = Activation(
                    'sigmoid', name=f'up_sigmoid_{i}_{num}_segnet')(output)
            else:
                output = Activation(
                    'relu', name=f'up_relu_{i}_{num}_segnet')(output)

            xs.append(output)

        return xs[-1]

    def create_model(self):
        self.input = Input(self.input_shape, name="Input_segnet")
        layer_1 = self.add_conv_layer(
            self.input, n_conv=2, n_filters=20, num=1)
        layer_2 = self.add_conv_layer(layer_1, n_conv=2, n_filters=40, num=2)
        layer_3 = self.add_conv_layer(layer_2, n_conv=3, n_filters=80, num=3)
        layer_4 = self.add_conv_layer(layer_3, n_conv=3, n_filters=100, num=4)

        deconv_3 = self.add_deconv_layer(
            layer_4, n_conv=3, n_filters=80, num=3)
        deconv_2 = self.add_deconv_layer(
            deconv_3, n_conv=3, n_filters=40, num=2)
        deconv_1 = self.add_deconv_layer(
            deconv_2, n_conv=2, n_filters=20, num=1)
        output = self.add_deconv_layer(
            deconv_1, n_conv=2, n_filters=1, last=True, num=0)

        model = Model(self.input, output)
        model.summary()

        opt = tf.keras.optimizers.Adam(lr=self.lr)
        model.compile(optimizer=opt, loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model


class UNet(getModel):
    def __init__(self, save_folder='./', lr=0.001, input_shape=(400, 400, 3), epochs=30, verbose=1,
                 batch_size=32, deepness=4, model_name='UNet.h5'):

        # Model specific params
        self.deepness = deepness
        self.lr = lr
        self.input_shape = input_shape
        getModel.__init__(self, save_folder, epochs,
                          verbose, batch_size, model_name)

        # Create and compile model UNet
        self.model = self.create_model()

    def create_model(self):
        self.input = keras.layers.Input(self.input_shape)
        self.skip = []
        inp = self.input

        # Convolution
        for x in range(self.deepness):
            filters = 2 ** (6 + x)
            skip, inp = self.conv_layer(filters, inp, num=x)
            self.skip.append(skip)

        # lowest layer
        conv1 = keras.layers.Conv2D(
            2 ** (6 + self.deepness), 3, activation='relu', padding='same', name='conv2d_1lowest_unet')(inp)
        conv2 = keras.layers.Conv2D(
            2 ** (6 + self.deepness), 3, activation='relu', padding='same', name=f'conv2d_2lowest_unet')(conv1)

        # Upsample and convolutions
        inp = conv2
        for x in range(self.deepness - 1, -1, -1):
            filters = 2 ** (6 + x)
            inp = self.upconv_layer(filters, inp, self.skip[x], num=x)

        output = keras.layers.Conv2D(1, 1, activation='sigmoid')(inp)
        model = keras.models.Model(inputs=self.input, outputs=output)
        model.summary()

        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr, ),
                      loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def conv_layer(self, filters, inp, num=0):
        dropout = SpatialDropout2D(0.1, name=f'down_dropout_{num}_unet')(inp)
        conv1 = keras.layers.Conv2D(
            filters, 3, activation='relu', padding='same', name=f'down_conv2d_1_{num}_unet')(dropout)
        conv2 = keras.layers.Conv2D(
            filters, 3, activation='relu', padding='same', name=f'down_conv2d_2_{num}_unet')(conv1)
        max_pool = keras.layers.MaxPool2D(
            2, strides=2, name=f'down_maxpool_{num}_unet')(conv2)
        return conv2, max_pool

    def upconv_layer(self, filters, inp, skip, num=0):
        dropout = SpatialDropout2D(0.1, name=f'up_dropout_{num}_unet')(inp)
        up_conv = keras.layers.Conv2DTranspose(
            filters, 2, 2, name=f'up_conv2dt_{num}_unet')(dropout)
        up_shape = up_conv.shape.as_list()
        skip_shape = skip.shape.as_list()

        x_start = (skip_shape[1] - up_shape[1]) // 2
        y_start = (skip_shape[2] - up_shape[2]) // 2
        x_end = x_start + up_shape[1]
        y_end = y_start + up_shape[2]

        cut_skip = keras.layers.Lambda(
            lambda x: x[:, x_start:x_end, y_start: y_end, :], name=f'up_lambda_{num}_unet')(skip)

        merge = keras.layers.Concatenate(
            axis=-1, name=f'up_merge_{num}_unet')([cut_skip, up_conv])
        conv1 = keras.layers.Conv2D(
            filters, 3, activation='relu', padding='same', name=f'up_conv2d_1_{num}_unet')(merge)
        conv2 = keras.layers.Conv2D(
            filters, 3, activation='relu', padding='same', name=f'up_conv2d_2_{num}_unet')(conv1)

        return conv2

    def set_params(self, **paramters):
        for paramter, value in paramters.items():
            setattr(self, paramter, value)


class ResUNet():
    def __init__(self, save_folder, input_shape=(400, 400, 3), epochs=30, verbose=1, batch_size=4, deepness=4):
        self.input_shape = input_shape
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.deepness = deepness
        self.save_folder = save_folder
        self.model_name = "res-unet"
        self.model_file = self.model_name + \
            datetime.datetime.today().strftime("_%d_%m_%y_%H:%M:%S") + ".h5"

    def res_block(self, x, nb_filters, strides):
        res_path = keras.layers.BatchNormalization()(x)
        res_path = keras.layers.Activation(activation='relu')(res_path)
        res_path = keras.layers.Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(
            res_path)
        res_path = keras.layers.BatchNormalization()(res_path)
        res_path = keras.layers.Activation(activation='relu')(res_path)
        res_path = keras.layers.Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(
            res_path)

        shortcut = keras.layers.Conv2D(
            nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
        shortcut = keras.layers.BatchNormalization()(shortcut)

        res_path = keras.layers.add([shortcut, res_path])
        return res_path

    def encoder(self, x):
        to_decoder = []

        main_path = keras.layers.Conv2D(filters=50, kernel_size=(
            3, 3), padding='same', strides=(1, 1))(x)
        main_path = keras.layers.BatchNormalization()(main_path)
        main_path = keras.layers.Activation(activation='relu')(main_path)

        main_path = keras.layers.Conv2D(filters=50, kernel_size=(
            3, 3), padding='same', strides=(1, 1))(main_path)

        shortcut = keras.layers.Conv2D(
            filters=50, kernel_size=(1, 1), strides=(1, 1))(x)
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
        main_path = keras.layers.concatenate(
            [main_path, from_encoder[2]], axis=3)
        main_path = self.res_block(main_path, [200, 200], [(1, 1), (1, 1)])

        main_path = keras.layers.UpSampling2D(size=(2, 2))(main_path)
        main_path = keras.layers.concatenate(
            [main_path, from_encoder[1]], axis=3)
        main_path = self.res_block(main_path, [100, 100], [(1, 1), (1, 1)])

        main_path = keras.layers.UpSampling2D(size=(2, 2))(main_path)
        main_path = keras.layers.concatenate(
            [main_path, from_encoder[0]], axis=3)
        main_path = self.res_block(main_path, [50, 50], [(1, 1), (1, 1)])

        return main_path

    def create_model(self):
        self.input = keras.layers.Input(shape=self.input_shape)

        to_decoder = self.encoder(x=self.input)

        path = self.res_block(x=to_decoder[2], nb_filters=[
                              400, 400], strides=[(2, 2), (1, 1)])
        path = self.decoder(x=path, from_encoder=to_decoder)
        path = keras.layers.Conv2D(filters=1, kernel_size=(
            1, 1), activation='sigmoid')(path)

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


class CombinedModel(getModel):
    '''
    This model combines an already saved UNet and an already saved SegNet and uses an Convolution
    to combined both models.
    '''

    def __init__(self, save_folder='./', lr=0.001, input_shape=(400, 400, 3), epochs=30, verbose=1,
                 batch_size=32, model_name='CombinedModel.h5'):
        self.lr = lr
        self.input_shape = input_shape
        getModel.__init__(self, save_folder, epochs,
                          verbose, batch_size, model_name)
        self.model = self.create_model()

    def train(self, X_train, Y_train, X_valid, Y_valid):

        #config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        #sess = tf.Session(config=config)
        # set this TensorFlow session as the default session for Keras
        # set_session(sess)

        # Train on the frozen models
        history = self.model.fit(x=(X_train, X_train), y=Y_train, validation_data=((X_valid, X_valid), Y_valid),
                                 batch_size=self.batch_size, verbose=self.verbose, epochs=self.epochs)

        print("\nTraining on frozen model finished\n")
        
        # Unfreeze all the layers 
        for layer in self.model.layers:
            layer.trainable = True
        
        self.model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),
                           loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train on the unfrozen layers
        history1 = self.model.fit(x=(X_train, X_train), y=Y_train, validation_data=((X_valid, X_valid), Y_valid),
                                  batch_size=self.batch_size, verbose=self.verbose, epochs=50)


        training_loss = history.history['loss']
        val_loss = history.history['val_loss']

        train_curves = {'train': training_loss, 'val': val_loss}

        with open(self.save_folder + 'train_curves.pickle', 'wb') as f:
            pickle.dump(train_curves, f)

        if not os.path.exists(self.save_folder + 'checkpoint/'):
            os.makedirs(self.save_folder + 'checkpoint')

        fileName = self.save_folder + 'checkpoint/' + self.model_name
        tf.keras.models.save_model(self.model, filepath=fileName)

    def predict(self, X):
        fileName = self.save_folder + 'checkpoint/' + self.model_name
        if not os.path.isfile(fileName):
            print("Model not found! Exiting ...")
            sys.exit(1)
        self.model = tf.keras.models.load_model(fileName)
        y_pred = self.model.predict((X, X), batch_size=self.batch_size)
        y_pred = (y_pred >= 0.5).astype(np.int)

        return y_pred

    def create_model(self):
        segnet = load_model('./SegNet/checkpoint/SegNet.h5')
        unet = load_model('./UNet/checkpoint/UNet.h5')

        # make the layers untrainable
        for layer in segnet.layers:
            layer.trainable = False

        for layer in unet.layers:
            layer.trainable = False

        # get ouput layers
        segnet_output = segnet.layers[-1].output
        unet_output = unet.layers[-1].output

        # Merge layers and apply  2D Convolutions
        concat = Concatenate(axis=-1)([segnet_output, unet_output])
        conv2d = Conv2D(1, (16, 16), padding='same',
                        activation='relu', name="conv2d_last_combined")(concat)
        outputs = conv2d

        # get inputs
        unet_input = unet.inputs[0]
        segnet_input = segnet.inputs[0]

        model = Model(inputs=[unet_input, segnet_input], outputs=outputs)
        model.summary()

        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),
                      loss='binary_crossentropy', metrics=['accuracy'])

        return model


class BaseLine(getModel):
    def __init__(self, save_folder='./', lr=0.001, input_shape=(400, 400, 3), epochs=30, verbose=1,
                 batch_size=32, deepness=4, model_name='BaselineUNet.h5'):

        # Model specific params
        self.deepness = deepness
        self.lr = lr
        self.input_shape = input_shape
        getModel.__init__(self, save_folder, epochs, 
                          verbose, batch_size, model_name)

        # Create and compile model UNet
        self.model = self.create_model()

    def create_model(self):
        '''
        This models is basically unet without any skip connections
        '''
        self.input = Input(self.input_shape)

        # Convolve downwards
        inp = self.input
        for x in range(self.deepness):
            filters = 2 ** (6 + 3)
            conv1 = keras.layers.Conv2D(
                filters, 3, activation='relu', padding='same', name=f'down_conv2d_1_{x}_baseline_unet'
            )(inp)
            conv2 = keras.layers.Conv2D(
                filters, 3, activation='relu', padding='same', name=f'down_conv2d_2_{x}_baseline_unet'
            )(conv1)
            max_pool = keras.layers.MaxPool2D(
                2, strides=2, name=f'down_maxpool_{x}_baseline_unet'
            )(conv2)
            inp = max_pool

        # Lowest layer convolution
        conv1 = keras.layers.Conv2D(
            2 ** (6 + self.deepness), 3, activation='relu', padding='same', name=f'conv2d_1_lowest_baseline_unet'
        )(inp)
        conv2 = keras.layers.Conv2D(
            2 ** (6 + self.deepness), 3, activation='relu', padding='same', name=f'conv2d_2_lowest_baseline_unet'
        )(conv2)

        # Convolve upwards
        inp = conv2
        for x in range(self.deepness -1 , -1 , -1):
            filters = 2 ** (6 + 3)
            up_conv = keras.layers.Conv2DTranspose(
                filters, 2, 2, name=f'up_conv2t_{x}_baseline_unet'
            )(inp)
            conv1 = keras.layers.Conv2D(
                filters, 3, activation='relu', padding='same', name=f'up_conv2dt_1_{x}_baseline_unet'
            )(up_conv)
            conv2 = keras.layers.Conv2D(
                filters, 3, activation='relu', padding='same', name=f'up_conv2dt_2_{x}_baseline_unet'
            )(conv1)
            inp = conv2

        # Create model
        output = keras.layers.Conv2D(1, 1, activation='sigmoid')(inp)
        model = keras.models.Model(inputs=self.input, outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),
            loss='binary_crossentropy', metrics=['accuracy']
        )
        return model


if __name__ == '__main__':
    # Just to verify proper behavior

    X_train = np.random.normal(size=(10, 400, 400, 3))
    Y_train = (np.random.normal(size=(10, 400, 400, 1)) >= 1).astype(np.int)

    X_valid = np.random.normal(size=(10, 400, 400, 3))
    Y_valid = (np.random.normal(size=(10, 400, 400, 1)) >= 1).astype(np.int)

    model = UNet(save_folder='./trial/', epochs=1)
    model.train(X_train, Y_train, X_valid, Y_valid)
    y_pred = model.predict(X_valid)
    print(y_pred.shape)

    model = SegNet(save_folder='./trial/', epochs=1)

    model.train(X_train, Y_train, X_valid, Y_valid)
    y_pred = model.predict(X_valid)
    print(y_pred.shape)

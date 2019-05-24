from tensorflow import keras
import numpy as np
import os
import sys
import tensorflow as tf
import h5py
from tensorflow.python.keras.layers import Dense,Input,Conv2D,BatchNormalization
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2DTranspose
import pickle

class getModel(object):
    def __init__(self, save_folder = './',epochs=30, verbose=1, batch_size=4,model_name = 'UNet.h5'):

        #Training specific params
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.save_folder = save_folder
        self.model_name = model_name

    def train(self, X_train, Y_train, X_valid, Y_valid):
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
        tf.keras.models.save_model(self.model,filepath=fileName)

        y_valid = self.model.predict(X_valid)
        y_valid = np.argmax(y_valid,axis=-1)
        with h5py.File(self.save_folder + 'validation.h5', 'w') as f:
            f['data'] = y_valid

    def predict(self, X):
        fileName = self.save_folder + 'checkpoint/' + self.model_name
        if not os.path.isfile(fileName):
            print("Model not found! Exiting ...")
            sys.exit(1)
        self.model = tf.keras.models.load_model(fileName)
        y_pred = self.model.predict(X, batch_size=self.batch_size)
        y_pred = np.argmax(y_pred,axis=-1)

        return y_pred

class SegNet(getModel):
    def __init__(self,save_folder = './',lr=0.001,input_shape = (400, 400, 3), epochs = 30, verbose=1,batch_size=4,model_name = 'SegNet.h5'):

        # Model specific params
        self.lr = lr
        self.input_shape = input_shape
        getModel.__init__(self,save_folder, epochs, verbose, batch_size,model_name)

        #Create and compile the model SegNet
        self.model = self.create_model()

    def add_conv_layer(self,input,n_filters,flt_sz=3,stride=1,n_conv=2):

        xs = [input]
        for i in range(n_conv):

            output = Conv2D(filters=n_filters,kernel_size=(flt_sz,flt_sz),strides=stride,\
                            padding="same")(xs[-1])

            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            xs.append(output)

        output = MaxPool2D()(xs[-1])

        return output

    def add_deconv_layer(self,input,n_filters,flt_sz=3,strides=2,n_conv=3,last=False):

        xs = [input]
        output = Conv2DTranspose(filters=n_filters, kernel_size=flt_sz, \
                                   strides=strides, padding='same')(input)
        xs.append(output)

        for i in range(n_conv):
            output = Conv2D(filters=n_filters, kernel_size=(flt_sz, flt_sz), strides=1, \
                            padding="same")(xs[-1])
            output = BatchNormalization()(output)

            if (last and i==n_conv-1):
                output = Activation('sigmoid')(output)
            else:
                output = Activation('relu')(output)

            xs.append(output)

        return xs[-1]

    def create_model(self):
        self.input = Input(self.input_shape)
        layer_1 = self.add_conv_layer(self.input, 20)
        layer_2 = self.add_conv_layer(layer_1, 40)
        layer_3 = self.add_conv_layer(layer_2, 80)

        deconv_1 = self.add_deconv_layer(layer_3,40)
        deconv_2 = self.add_deconv_layer(deconv_1,20)
        output = self.add_deconv_layer(deconv_2,1,last=True)

        model = Model(self.input,output)
        model.summary()

        opt = tf.keras.optimizers.Adam(lr = self.lr)
        binary_loss = tf.keras.losses.binary_crossentropy
        model.compile(optimizer = opt,loss = binary_loss,metrics = ['accuracy'])

        return model

class UNet(getModel):
    def __init__(self, save_folder = './', lr=0.001, input_shape=(400, 400, 3), epochs=30, verbose=1, batch_size=4,
                 deepness=4,model_name = 'Unet.h5'):

        #Model specific params
        self.deepness = deepness
        self.lr = lr
        self.input_shape = input_shape
        getModel.__init__(self,save_folder, epochs, verbose, batch_size,model_name)

        #Create and compile model UNet
        self.model = self.create_model()


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
            2**(6 + self.deepness), 3, activation='relu', padding='same')(inp)
        conv2 = keras.layers.Conv2D(
            2**(6 + self.deepness), 3, activation='relu', padding='same')(conv1)

        # Upsample and convolutions
        inp = conv2
        for x in range(self.deepness - 1, -1, -1):
            filters = 2**(6 + x)
            inp = self.upconv_layer(filters, inp, self.skip[x])

        output = keras.layers.Conv2D(1, 1, activation='sigmoid')(inp)
        model = keras.models.Model(inputs=self.input, outputs=output)
        model.summary()

        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr, ),
                           loss='binary_crossentropy', metrics=['accuracy'])

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

    def set_params(self, **paramters):
        for paramter, value in paramters.items():
            setattr(self, paramter, value)

if __name__ == '__main__':
    #Just to verify proper behavior

    X_train = np.random.normal(size=(10, 400, 400, 3))
    Y_train = (np.random.normal(size=(10, 400, 400, 1)) >= 1).astype(np.int)

    X_valid = np.random.normal(size=(10, 400, 400, 3))
    Y_valid = (np.random.normal(size=(10, 400, 400, 1)) >= 1).astype(np.int)

    model = UNet(save_folder='./trial/', epochs=1)
    model.train(X_train, Y_train, X_valid, Y_valid)
    y_pred = model.predict(X_valid)
    print(y_pred.shape)

    model = SegNet(save_folder='./trial/',epochs=1)


    model.train(X_train,Y_train,X_valid,Y_valid)
    y_pred = model.predict(X_valid)
    print(y_pred.shape)


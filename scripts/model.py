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
import matplotlib.pyplot as plt

class SegNet():
    def __init__(self,im_sz=32,n_channels=3,lr=0.001,n_epochs=100,save_folder='SegNet/',batch_sz=32):
        self.im_sz = im_sz
        self.n_channels = n_channels
        self.lr = lr
        self.n_epochs = n_epochs
        self.save_folder = save_folder
        self.batch_sz = batch_sz
        self.main_input = Input(shape=[im_sz,im_sz,n_channels])


        #Create the model
        self.model = self.create()


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
                output = Activation('softmax')(output)
            else:
                output = Activation('relu')(output)

            xs.append(output)

        return xs[-1]

    def create(self):
        layer_1 = self.add_conv_layer(self.main_input, 20)
        layer_2 = self.add_conv_layer(layer_1, 40)
        layer_3 = self.add_conv_layer(layer_2, 80)

        deconv_1 = self.add_deconv_layer(layer_3,40)
        deconv_2 = self.add_deconv_layer(deconv_1,20)
        self.main_output = self.add_deconv_layer(deconv_2,1,last=True)

        model = Model(self.main_input,self.main_output)

        opt = tf.keras.optimizers.Adam(lr = self.lr)
        binary_loss = tf.keras.losses.binary_crossentropy
        model.compile(optimizer = opt,loss = binary_loss)

        return model

    def train(self,X_train,y_train,X_valid,y_valid):
        model = self.model
        history = model.fit(X_train,y_train,validation_data = (X_valid,y_valid),\
                  epochs=self.n_epochs,batch_size = self.batch_sz)


        training_loss = history.history['loss']
        val_loss = history.history['val_loss']

        train_curves = {'train':training_loss,'val':val_loss}

        with open(self.save_folder + 'train_curves.pickle','wb') as f:
            pickle.dump(train_curves,f)

        if not os.path.exists(self.save_folder + 'checkpoint'):
            os.makedirs(self.save_folder + 'checkpoint')

        fileName = self.save_folder + 'checkpoint/SegNet.h5'

        tf.keras.models.save_model(model,filepath = fileName)

    def predict(self,X_test,test_idxs):
        fileName = self.save_folder + 'checkpoint/SegNet.h5'
        if not os.path.isfile(fileName):
            print("Model not found! Exiting ...")
            sys.exit(1)

        model = tf.keras.models.load_model(fileName)
        y_pred = model.predict(X_test)
        y_pred = (y_pred>=0.5).astype(np.int)


        if not os.path.exists(self.save_folder + 'pred_imgs/'):
            os.makedirs(self.save_folder + 'pred_imgs/')

        for i in range(y_pred.shape[0]):
            plt.imsave(self.save_folder + 'pred_imgs/img_{}.png'.format(test_idxs[i]))

        with h5py.File(self.save_folder + 'predictions.h5','w') as f:
            f['data'] = y_pred

        return y_pred
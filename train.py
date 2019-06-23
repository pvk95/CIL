from utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow._api.v1 import keras
from tensorflow._api.v1.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


class Metrics(keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()

        self.validation_data = valid_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')
        preds = np.asarray(self.model.predict(self.validation_data[0]))
        val_predict = np.argmax(preds, axis=-1)
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: % f — val_precision: % f — val_recall % f" %
              (_val_f1, _val_precision, _val_recall))


PATCH_SIZE = 16

inp = keras.Input(shape=(16, 16, 3))

x = keras.layers.Conv2D(64, 3)(inp)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation("relu")(x)
x = keras.layers.MaxPool2D()(x)

x = keras.layers.Conv2D(128, 3)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation("relu")(x)
x = keras.layers.MaxPool2D()(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(2, activation='softmax')(x)

model = keras.Model(inp, x, name='basic_cnn')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=["acc"])
model.summary()

X, y = load_data(PATCH_SIZE, PATCH_SIZE)
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      test_size=0.1,
                                                      shuffle=True)

metrics = Metrics(valid_data=(X_valid, y_valid))
hist = model.fit(X_train, y_train,
                 batch_size=32,
                 epochs=20,
                 validation_data=(X_valid, y_valid),
                 callbacks=[metrics])

# orig, rec = reconstruct_gt(22, PATCH_SIZE, PATCH_SIZE, plot=True)
# plt.imshow(X[22])

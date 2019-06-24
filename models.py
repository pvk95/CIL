from tensorflow._api.v1 import keras
import tensorflow as tf
from utils import *
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight


class Base(object):
    def __init__(self, config=None, model_name="Base"):
        self.name = model_name
        self.metrics = None
        self.model = None
        self.history = None
        self.config = config

        if self.config is None:
            self.config = Config()

    def train(self, X_train, Y_train, X_valid, Y_valid):
        self.metrics = F1(valid_data=(X_valid, Y_valid),
                          patience=self.config.patience)

        class_weight = None
        if self.config.use_class_weights:
            class_weight = compute_class_weight('balanced',
                                                np.unique(Y_train),
                                                Y_train)
            print("Using class weights for training: %s" % class_weight)

        self.history = self.model.fit(X_train, Y_train,
                                      batch_size=self.config.batch_size,
                                      epochs=self.config.epochs,
                                      validation_data=(X_valid, Y_valid),
                                      callbacks=[self.metrics],
                                      class_weight=class_weight)

    def plots(self, savefig=False):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        f1 = self.metrics.val_f1s
        recall = self.metrics.val_recalls
        precision = self.metrics.val_precisions

        epochs = range(1, len(loss) + 1)

        plt.clf()
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        # Training and validation loss
        axs[0, 0].plot(epochs, loss, 'b--', label='Training loss')
        axs[0, 0].plot(epochs, val_loss, 'b', label='Validation loss')
        axs[0, 0].set_title('Training and validation loss')
        axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()

        # Training and validation accuracy
        axs[0, 1].plot(epochs, acc, 'b--', label='Training acc')
        axs[0, 1].plot(epochs, val_acc, 'b', label='Validation acc')
        axs[0, 1].set_title('Training and validation accuracy')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].legend()

        # F1-Score
        axs[1, 0].plot(epochs, f1, 'b', label='Validation F1-Score')
        axs[1, 0].set_title('Validation F1-Score')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('F1-Score')
        axs[1, 0].legend()

        # Recall and Precision
        axs[1, 1].plot(epochs, recall, 'b--', label='Recall')
        axs[1, 1].plot(epochs, precision, 'b', label='Precision')
        axs[1, 1].set_title('Validation recall and precision score')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].legend()

        # plt.show()
        if savefig:
            time = datetime.now().strftime("%Y%m%d-%H%M%S")
            fn = self.config.plots_dir + self.name + "_" + time + ".png"
            fig.savefig(fn)

        return fig


class BasicCNN(Base):
    def __init__(self, config=None, model_name='BasicCNN'):
        super(BasicCNN, self).__init__(config=config, model_name=model_name)

        self.model = self.build_model()

    def build_model(self):
        inp = keras.Input(shape=(self.config.patch_height,
                                 self.config.patch_width, 3))

        x = keras.layers.Conv2D(8, 3)(inp)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)

        x = keras.layers.Conv2D(16, 3)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = keras.layers.Dense(2, activation='softmax')(x)

        model = keras.Model(inp, x, name=self.name)
        model.compile(optimizer=self.config.optimizer,
                      loss=self.config.loss,
                      metrics=["acc"])
        return model

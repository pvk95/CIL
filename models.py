from tensorflow._api.v1 import keras
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from utils import *
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
import os
import json


class Base(object):
    def __init__(self, config=None, model_name="Base"):
        self.name = model_name
        self.metrics = None
        self.model = None
        self.history = None
        self.config = config
        self.model_dir = None

        if self.config is None:
            self.config = Config()

        self.init_run()

    def init_run(self):
        if not os.path.exists(self.config.runs_dir):
            os.mkdir(self.config.runs_dir)
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir_name = self.name + "_" + time
        self.model_dir = os.path.abspath(os.path.join(self.config.runs_dir,
                                                      model_dir_name))
        print("Writing to %s\n" % self.model_dir)
        os.mkdir(self.model_dir)
        json.dump(self.config.__dict__,
                  open(os.path.join(self.model_dir, 'config.json'), 'w'),
                  indent=4,
                  sort_keys=True)

    def log_model(self):
        # log model.summary
        with open(os.path.join(self.model_dir, "summary.txt"), 'w') as fh:
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

        # plot model image
        plot_model(self.model,
                   os.path.join(self.model_dir, "model.png"),
                   show_shapes=True)

    def train(self, X_train, Y_train, X_valid, Y_valid):
        self.log_model()
        self.metrics = F1(valid_data=(X_valid, Y_valid),
                          patience=self.config.patience)

        if self.config.use_class_weights:
            class_weight = compute_class_weight('balanced',
                                                np.unique(Y_train),
                                                Y_train)
            print("Using class weights for training: %s" % class_weight)
        else:
            class_weight = None

        self.history = self.model.fit(X_train, Y_train,
                                      batch_size=self.config.batch_size,
                                      epochs=self.config.epochs,
                                      validation_data=(X_valid, Y_valid),
                                      callbacks=[self.metrics],
                                      class_weight=class_weight)

        # save plots
        _ = self.plots(savefig=True)

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
        idx = f1.index(max(f1))
        axs[1, 0].plot(epochs, f1, 'b', label='Validation F1-Score')
        axs[1, 0].set_title(
            'Validation F1-Score:%.2f (acc:%.2f)'
            % (f1[idx], val_acc[idx]))
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
            fn = os.path.join(self.model_dir, 'plots.png')
            fig.savefig(fn)

        return fig


class BasicCNN(Base):
    def __init__(self, config=None, model_name='BasicCNN'):
        super(BasicCNN, self).__init__(config=config, model_name=model_name)

        self.model = self.build_model()

    def build_model(self):
        inp = keras.Input(shape=(self.config.patch_height,
                                 self.config.patch_width, 3))

        x = keras.layers.Conv2D(8, 3, kernel_initializer='glorot_normal')(inp)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)

        x = keras.layers.Conv2D(16, 3, kernel_initializer='glorot_normal')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D()(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(rate=0.3)(x)
        x = keras.layers.Dense(2, activation='softmax',
                               kernel_initializer='glorot_normal')(x)

        model = keras.Model(inp, x, name=self.name)
        model.compile(optimizer=self.config.optimizer,
                      loss=self.config.loss,
                      metrics=["acc"])
        return model

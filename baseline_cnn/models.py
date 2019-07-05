from tensorflow._api.v1 import keras
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from utils import *
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
import os
import json
from PIL import Image


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

        print("Start training")
        self.history = self.model.fit(X_train, Y_train,
                                      batch_size=self.config.batch_size,
                                      epochs=self.config.epochs,
                                      validation_data=(X_valid, Y_valid),
                                      callbacks=[self.metrics],
                                      class_weight=class_weight)

        _ = self.plots(savefig=True)

    def predict(self, X, file_names, img_sz=608, sz=16):
        y_pred = self.model.predict(X)
        y_pred = np.argmax(y_pred, axis=-1)
        image_preds = patches2images(y_pred, img_sz, sz)
        # Save predictions
        dir_path = os.path.join(self.model_dir, self.config.pred_dir)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        for i in range(0, image_preds.shape[0]):
            file = os.path.join(dir_path, file_names[i])
            # convert to rgb
            rgb = image_preds[i] - np.min(image_preds[i])
            rgb = (rgb / np.max(rgb) * 255).round().astype(np.uint8)
            im = Image.fromarray(rgb)
            im.save(file)
        return image_preds

    def plots(self, savefig=False):
        print("Saving training plots")
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
        print("Log model data")
        print("\tLog model summary")
        with open(os.path.join(self.model_dir, "summary.txt"), 'w') as fh:
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

        print("\tLog model image")
        plot_model(self.model,
                   os.path.join(self.model_dir, "model.png"),
                   show_shapes=True)


class BasicCNN(Base):
    def __init__(self, config=None, model_name='BasicCNN'):
        super(BasicCNN, self).__init__(config=config, model_name=model_name)

        self.model = self.build_model()

    def build_model(self):
        inp = keras.Input(shape=(self.config.patch_height,
                                 self.config.patch_width, 3))

        x = keras.layers.Conv2D(8, 3, kernel_initializer='glorot_normal')(inp)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.PReLU()(x)
        x = keras.layers.MaxPool2D()(x)

        x = keras.layers.Conv2D(16, 3, kernel_initializer='glorot_normal')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.PReLU()(x)
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


class BasicFCN(Base):
    def __init__(self, config=None, model_name='BasicFCN'):
        super(BasicFCN, self).__init__(config=config, model_name=model_name)

        self.model = self.build_model()

    def build_model(self):
        inp = keras.Input(shape=(self.config.patch_height,
                                 self.config.patch_width, 3))

        x = keras.layers.Conv2D(32, 5, padding="same",
                                kernel_initializer='glorot_normal')(inp)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.PReLU()(x)
        x = keras.layers.MaxPool2D()(x)

        x = keras.layers.Conv2D(64, 5, padding="same",
                                kernel_initializer='glorot_normal')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.PReLU()(x)
        x = keras.layers.MaxPool2D()(x)

        x = keras.layers.Conv2D(2, 1, kernel_initializer='glorot_normal',
                                activation='softmax')(x)
        x = keras.layers.GlobalAveragePooling2D()(x)

        model = keras.Model(inp, x, name=self.name)
        model.compile(optimizer=self.config.optimizer,
                      loss=self.config.loss,
                      metrics=["acc"])
        return model


class F1(keras.callbacks.Callback):
    """Callback that computes the F1-score, precision and recall on validation data and stops training when the F1-score stops improving.
    """

    def __init__(self, valid_data, patience=0, restore_best_weights=True, min_delta=5e-4):
        super(F1, self).__init__()

        self.validation_data = valid_data
        # Early Stopping
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        # self.val_accuracies = []
        # self.val_losses = []

        # Early Stopping
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')

        preds = np.asarray(self.model.predict(self.validation_data[0]))
        val_predict = np.argmax(preds, axis=-1)
        val_targ = self.validation_data[1]

        average = 'macro'
        _val_f1 = metrics.f1_score(val_targ, val_predict, average=average)
        _val_recall = metrics.recall_score(val_targ,
                                           val_predict,
                                           average=average)
        _val_precision = metrics.precision_score(val_targ,
                                                 val_predict,
                                                 average=average)
        # _val_accuracy = metrics.accuracy_score(val_targ, val_predict)
        # _val_loss = metrics.log_loss(val_targ, preds)

        # store in instance
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        # self.val_accuracies.append(_val_accuracy)
        # self.val_losses.append(_val_loss)
        print(" - val_f1: % f - val_precision: % f - val_recall % f" %
              (_val_f1, _val_precision, _val_recall))

        # Early Stopping
        current = _val_f1
        if np.greater(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    print('Early Stopping: Restoring model weights \
                        from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping. Best F1-Score=%f' %
                  (self.stopped_epoch + 1, self.best))

from tensorflow._api.v1.keras.models import Model
from tensorflow._api.v1.keras.models import load_model
from tensorflow._api.v1.keras.layers import (
    Dropout,
    Activation,
    PReLU,
    Softmax,
    ReLU,
    LeakyReLU,
)
from tensorflow._api.v1.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow._api.v1.keras.layers import Add
from tensorflow._api.v1.keras.applications import vgg16
from tensorflow._api.v1.keras import optimizers
from tensorflow._api.v1.keras.utils import Sequence
from tensorflow._api.v1.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow._api.v1.keras import backend as K
import tensorflow as tf
from tensorflow._api.v1.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from shutil import copyfile
import numpy as np
from sklearn.utils import shuffle
from imgaug import augmenters as iaa


class FCN8:
    def __init__(
        self,
        name="FCN8",
        dropout=0.5,
        batch_size=1,
        epochs=100,
        optimizer=None,
        loss="categorical_crossentropy",
        patience=11,  # early stopping
        restore=None,  # chkpt folder name
        baseline=None,  # early stopping
        use_multiprocessing=False,  # data gen
        workers=4,  # data generator
        tune_level=None,
    ):
        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.runs_dir = "runs"
        self.pred_dir = "predictions"
        self.restore = restore
        self.patience = patience
        self.baseline = baseline
        self.dropout = dropout
        self.multiprocess = use_multiprocessing
        self.workers = workers
        self.X = None
        self.Y = None
        self.tune_count = 0
        self.init_run()
        # has to be after init_run(json.dump cannot serialize this)
        if optimizer is None:
            self.optimizer = optimizers.SGD(
                decay=5 ** (-4), momentum=0.9, nesterov=True
            )
        else:
            self.optimizer = optimizer

        if restore is None:
            print("Building model\n")
            self.build()
        else:
            checkpoint_name = "model-" + self.name + ".hdf5"
            src = os.path.join(self.runs_dir, self.restore, checkpoint_name)
            print("Restoring model from %s\n" % src)
            self.model = load_model(src, custom_objects={"iou": self.iou})

        if not tune_level is None:
            print("Freezing encoder\n")
            self.__freeze(tune_level=tune_level)

        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=["acc", self.iou]
        )

    def __freeze(self, tune_level=0):
        # 0=all vgg16 layers frozen
        # 5=all vgg16 layers unfrozen
        layer_names = []
        for i in range(1, 6 - tune_level):
            layer_names.append("block%d" % i)

        for layer in self.model.layers:
            if layer.name.startswith("block"):
                print("unfreezing %s" % layer.name)
                layer.trainable = True
            for name in layer_names:
                if layer.name.startswith(name):
                    print("freezing %s" % layer.name)
                    layer.trainable = False

    def fine_tune(
        self,
        tune_level=1,
        optimizer=None,
        lr=1e-4,
        epochs=100,
        test_size=0.1,
        shuffle=True,
    ):
        # Freezing all layers up to a specific one
        if self.X is None:
            print("Cannot fine-tune. Must train the model first\n")
            return

        self.epochs = epochs
        self.__freeze(tune_level=tune_level)
        self.optimizer = optimizer

        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=["acc", self.iou]
        )
        self.model.summary()
        print("Model recompiled with lr=%f!\n" % lr)
        print("Restarting training...\n")
        self.tune_count += 1
        self.train(self.X, self.Y, test_size=test_size, shuffle=shuffle, tuning=True)

    def init_run(self):
        # setup directory for current run and dump the configuration of the model to folder
        if not os.path.exists(self.runs_dir):
            os.mkdir(self.runs_dir)
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir_name = self.name + "_" + time
        self.model_dir = os.path.abspath(os.path.join(self.runs_dir, model_dir_name))
        print("Writing to %s\n" % self.model_dir)
        os.mkdir(self.model_dir)
        json.dump(
            self.__dict__,
            open(os.path.join(self.model_dir, "fcn8_config.json"), "w"),
            skipkeys=True,
            indent=4,
            sort_keys=True,
        )

    def log_model(self):
        # log model graph
        print("Log model data")
        print("Log model summary")
        with open(os.path.join(self.model_dir, "summary.txt"), "w") as fh:
            self.model.summary(print_fn=lambda x: fh.write(x + "\n"))

        print("Log model image\n")
        plot_model(
            self.model, os.path.join(self.model_dir, "model.png"), show_shapes=True
        )

    def build(self):
        # build the fcn8 graph
        self.vgg16_enc()
        self.model = self.fcn(
            2, self.layers, self.vgg16_model.input, dropout=self.dropout
        )

    def train(self, X, Y, test_size=0.1, shuffle=True, tuning=False, seed=1234):
        # train the model
        self.X = X
        self.Y = Y
        if not tuning:  # check if fine tuning
            self.log_model()

        X_train, X_valid, Y_train, Y_valid = train_test_split(
            self.X, self.Y, test_size=test_size, shuffle=shuffle, random_state=seed
        )
        trainGenerator = DataGen(
            X_train, Y_train, batch_size=self.batch_size, train=True, shuffle=True
        )
        validGenerator = DataGen(
            X_valid, Y_valid, batch_size=self.batch_size, train=False, shuffle=False
        )

        reduce_lr_on_plateau_sgd = ReduceLROnPlateau(
            monitor="val_iou",
            factor=0.6,
            patience=5,
            verbose=1,
            mode="max",
            min_delta=1e-4,
            cooldown=0,
            min_lr=1e-8,
        )

        early_stopping_sgd = EarlyStopping(
            monitor="val_iou",
            min_delta=1e-4,
            patience=self.patience,
            verbose=1,
            mode="max",
            baseline=self.baseline,
            restore_best_weights=True,
        )

        tensorboard = TensorBoard(
            log_dir=os.path.join(self.model_dir, "logs"),
            histogram_freq=0,
            # batch_size=self.batch_size,
            write_graph=True,
            write_grads=True,
            write_images=True,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None,
        )

        checkpoint_name = "model-" + self.name + ".hdf5"
        checkpoint_cb = ModelCheckpoint(
            filepath=os.path.join(self.model_dir, checkpoint_name),
            monitor="val_iou",
            mode="max",
            verbose=1,
            save_best_only=True,
            period=1,
        )

        model_callbacks_sgd = [
            tensorboard,
            checkpoint_cb,
            reduce_lr_on_plateau_sgd,
            early_stopping_sgd,
        ]

        self.history = self.model.fit_generator(
            trainGenerator,
            validation_data=validGenerator,
            use_multiprocessing=self.multiprocess,
            workers=self.workers,
            epochs=self.epochs,
            callbacks=model_callbacks_sgd,
        )

        _ = self.plots(savefig=True, tuning=tuning)

    def plots(self, savefig=False, tuning=False):
        # Save plots of training/validation loss and metrics
        print("Saving training plots")
        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        acc = self.history.history["acc"]
        val_acc = self.history.history["val_acc"]
        iou = self.history.history["iou"]
        val_iou = self.history.history["val_iou"]

        epochs = range(1, len(loss) + 1)

        gs = gridspec.GridSpec(2, 2)
        fig = pl.figure(figsize=(10, 10))

        ax = pl.subplot(gs[0, 0])  # row 0, col 0
        pl.plot(epochs, loss, "b--", label="Training loss")
        pl.plot(epochs, val_loss, "b", label="Validation loss")
        ax.set_title("Training and validation loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()

        ax = pl.subplot(gs[0, 1])  # row 0, col 1
        pl.plot(epochs, acc, "b--", label="Training acc")
        pl.plot(epochs, val_acc, "b", label="Validation acc")
        ax.set_title("Training and validation accuracy")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()

        idx = val_iou.index(max(val_iou))
        ax = pl.subplot(gs[1, :])  # row 1, span all columns
        ax.plot(epochs, iou, "b--", label="Training IoU")
        ax.plot(epochs, val_iou, "b", label="Validation IoU")
        ax.set_title(
            "Training and validation IoU. val_iou:%.4f val_acc:%.4f"
            % (val_iou[idx], val_acc[idx])
        )
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()

        # plt.show()
        if savefig:
            if tuning:
                fn = os.path.join(
                    self.model_dir, "plots_tuning_%d.png" % self.tune_count
                )
            else:
                fn = os.path.join(self.model_dir, "plots.png")
            fig.savefig(fn)

        return fig

    def vgg16_enc(self, include_top=False):
        # build the vGG16 graph and extract the layers needed for fcn8
        self.vgg16_model = vgg16.VGG16(
            include_top=include_top,
            weights="imagenet",
            input_tensor=None,
            input_shape=(608, 608, 3),
            pooling=None,
            classes=1000,
        )

        layer1 = self.vgg16_model.get_layer("block1_pool").output
        layer2 = self.vgg16_model.get_layer("block2_pool").output
        layer3 = self.vgg16_model.get_layer("block3_pool").output
        layer4 = self.vgg16_model.get_layer("block4_pool").output
        layer5 = self.vgg16_model.get_layer("block5_pool").output
        self.layers = [layer1, layer2, layer3, layer4, layer5]

    def fcn(self, num_class, feature_layers, model_input, dropout=0.5):
        # build the actual fcn8 graph

        # n = 4096
        n = 1024

        pool5 = feature_layers[-1]  # pool5
        pool4 = feature_layers[-2]  # pool 4
        pool3 = feature_layers[-3]  # pool 3

        # fc6
        fc6 = Conv2D(
            n, 7, padding="same", kernel_initializer="glorot_normal", name="fc6"
        )(pool5)
        fc6 = BatchNormalization()(fc6)
        fc6 = LeakyReLU()(fc6)
        fc6 = Dropout(dropout, name="fc6_dropout")(fc6)

        # fc7
        fc7 = Conv2D(
            n, 1, padding="same", kernel_initializer="glorot_normal", name="fc7"
        )(fc6)
        fc7 = BatchNormalization()(fc7)
        fc7 = LeakyReLU()(fc7)
        fc7 = Dropout(dropout, name="fc7_dropout")(fc7)

        # unpool
        fc7_4up = Conv2DTranspose(
            num_class,
            kernel_size=4,
            strides=4,
            use_bias=False,
            kernel_initializer="glorot_normal",
            # output_padding=(2, 2),  # for 400px
            name="fc7_4up",
        )(fc7)

        # unpool
        pool4_pred = Conv2D(
            num_class,
            1,
            padding="same",
            kernel_initializer="glorot_normal",
            name="pool4_pred",
        )(pool4)
        pool4_pred = BatchNormalization()(pool4_pred)
        pool4_pred = LeakyReLU()(pool4_pred)
        pool4_2up = Conv2DTranspose(
            num_class,
            kernel_size=2,
            strides=2,
            use_bias=False,
            name="pool4_up",
            kernel_initializer="glorot_normal",
        )(pool4_pred)

        # pool3 prediction
        pool3_pred = Conv2D(
            num_class,
            1,
            padding="same",
            kernel_initializer="glorot_normal",
            name="pool3_pred",
        )(pool3)
        pool3_pred = BatchNormalization()(pool3_pred)
        pool3_pred = LeakyReLU()(pool3_pred)

        # Unpool
        out = Add(name="fuse")([pool4_2up, pool3_pred, fc7_4up])
        out = Conv2DTranspose(
            num_class,
            kernel_size=8,
            strides=8,
            use_bias=False,
            name="upsample",
            kernel_initializer="glorot_normal",
        )(out)
        out = Softmax()(out)

        model = Model(model_input, out)

        return model

    def iou(self, true, pred):
        # The metric used for early stopping and learning rate decrease.
        # source: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63044
        def castF(x):
            return K.cast(x, K.floatx())

        def castB(x):
            return K.cast(x, bool)

        def iou_loss_core(
            true, pred
        ):  # this can be used as a loss if you make it negative
            intersection = true * pred
            notTrue = 1 - true
            union = true + (notTrue * pred)

            return (K.sum(intersection, axis=-1) + K.epsilon()) / (
                K.sum(union, axis=-1) + K.epsilon()
            )

        def metric(true, pred):
            # metric function for keras.

            tresholds = [0.5 + (i * 0.05) for i in range(10)]

            # flattened images (batch, pixels)
            true = K.batch_flatten(true)
            pred = K.batch_flatten(pred)
            pred = castF(K.greater(pred, 0.5))

            # total white pixels - (batch,)
            trueSum = K.sum(true, axis=-1)
            predSum = K.sum(pred, axis=-1)

            # has mask or not per image - (batch,)
            true1 = castF(K.greater(trueSum, 1))
            pred1 = castF(K.greater(predSum, 1))

            # to get images that have mask in both true and pred
            truePositiveMask = castB(true1 * pred1)

            # separating only the possible true positives to check iou
            testTrue = tf.boolean_mask(true, truePositiveMask)
            testPred = tf.boolean_mask(pred, truePositiveMask)

            # getting iou and threshold comparisons
            iou = iou_loss_core(testTrue, testPred)
            truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

            # mean of thressholds for true positives and total sum
            truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
            truePositives = K.sum(truePositives)

            # to get images that don't have mask in both true and pred
            trueNegatives = (1 - true1) * (
                1 - pred1
            )  # = 1 -true1 - pred1 + true1*pred1
            trueNegatives = K.sum(trueNegatives)

            return (truePositives + trueNegatives) / castF(K.shape(true)[0])

        return metric(true, pred)


class DataGen(Sequence):
    # Generator for the training and validation images
    # Image augmentation only applied to training images
    def __init__(
        self, X, Y, batch_size=1, shape=(224, 224, 3), train=True, shuffle=True
    ):
        self.batch_size = batch_size
        self.Y = Y
        self.X = X
        self.shuffle = shuffle
        self.train = train

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        X_batch = self.X[index * self.batch_size : (index + 1) * self.batch_size]
        Y_batch = self.Y[index * self.batch_size : (index + 1) * self.batch_size]
        if self.train:
            return self.__augment(X_batch, Y_batch)
        else:
            return X_batch, Y_batch

    def on_epoch_end(self):
        if self.shuffle == True:
            self.X, self.Y = shuffle(self.X, self.Y)

    def __augment(self, X, Y):
        # Augments the current batch of training images and their ground truth
        affine = iaa.Affine(rotate=(-180, 180), scale=(0.9, 1.1), mode=["reflect"])

        seq_both = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Sometimes(0.5, affine),
                # iaa.Sometimes(0.5,iaa.Crop(px=(0, 100))),
            ],
            random_order=False,
        ).to_deterministic()

        Xaug = seq_both(images=X)
        Yaug = seq_both(images=Y)
        return Xaug, Yaug

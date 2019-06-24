import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn import metrics
from scripts.mask_to_submission import *
from tensorflow._api.v1 import keras
import tensorflow as tf


TRAIN_PATH = "data/training/"
TEST_PATH = "data/test_images/"
GT_PATH = TRAIN_PATH + "groundtruth/"
IMG_PATH = TRAIN_PATH + "images/"


class Config(object):
    def __init__(self,
                 batch_size=32,
                 epochs=100,
                 optimizer="adam",
                 loss="sparse_categorical_crossentropy",
                 plots_dir="plots/",
                 patience=0,
                 patch_width=16,
                 patch_height=16,
                 use_class_weights=True):
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.plots_dir = plots_dir
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patience = patience
        self.use_class_weights = use_class_weights


class F1(keras.callbacks.Callback):
    """Callback that computes the F1-score, precision and recall on validation data and stops training when the F1-score stops improving.
    """

    def __init__(self, valid_data, patience=0, restore_best_weights=False):
        super(F1, self).__init__()

        self.validation_data = valid_data
        # Early Stopping
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

        _val_f1 = metrics.f1_score(val_targ, val_predict, average='macro')
        _val_recall = metrics.recall_score(
            val_targ, val_predict, average='macro')
        _val_precision = metrics.precision_score(
            val_targ, val_predict, average='macro')
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
        if np.greater(current, self.best):
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
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


def reconstruct_gt(idx, w=16, h=16, plot=False):
    img_name = "satImage_%.3d.png" % idx
    fn = GT_PATH + img_name
    assert os.path.isfile(fn), \
        "Image file %s doesn't exist" % fn

    orig_img = mpimg.imread(fn)
    img_height = orig_img.shape[0]
    img_width = orig_img.shape[1]

    patches = image_patches(orig_img)
    labels = [patch_to_label(patch) for patch in patches]

    rec_img = image_from_labels(labels, img_height, img_width, h, w)

    if plot:
        fig, (orig_ax, rec_ax) = plt.subplots(1, 2, figsize=(10, 10))
        orig_ax.set_axis_off()
        orig_ax.set_title('Original Ground Truth')
        orig_ax.imshow(orig_img, cmap=plt.cm.binary_r)

        rec_ax.set_axis_off()
        rec_ax.set_title('Reconstructed Ground Truth')
        rec_ax.imshow(rec_img, cmap=plt.cm.binary_r)
        plt.show()

    return orig_img, rec_img


def image_from_labels(labels, img_height, img_width, w=16, h=16):
    img = np.zeros([img_width, img_height])
    idx = 0
    for i in range(0, img_height, h):
        for j in range(0, img_width, w):
            if labels[idx] > 0.5:  # road
                img[i:i+h, j:j+w] = 1
            else:
                img[i:i+h, j:j+w] = 0
            idx = idx + 1
    return img


def image_patches(img, h=16, w=16):
    list_patches = []
    img_height = img.shape[0]
    img_width = img.shape[1]
    for i in range(0, img_height, h):
        for j in range(0, img_width, w):
            if len(img.shape) < 3:
                im_patch = img[i:i+h, j:j+w]
            else:
                im_patch = img[i:i+h, j:j+w, :]
            list_patches.append(im_patch)
    return list_patches


def load_data(h=16, w=16):
    print("Loading data...")
    fn = [f for _, _, f in os.walk(IMG_PATH)][0]

    print("\tLoading images and ground truths")
    images, gts = [], []
    for f in fn:
        img = mpimg.imread(IMG_PATH + f)
        gt = mpimg.imread(GT_PATH + f)
        images.append(img)
        gts.append(gt)

    n = len(images)
    print("\tLoaded", n, "images and", len(gts), "ground truths")
    assert n == len(gts), "no. of images not equal to ground truths"

    img_height = images[0].shape[0]
    img_width = images[0].shape[1]
    no_image_patches = (img_height * img_width)//(h*w)
    print("\tImage dimensions WxH: %dx%d" % (img_width, img_height))
    print("\t%d patches of size %dx%d per image" %
          (no_image_patches, w, h))
    assert img_height == gts[0].shape[0] and img_width == gts[0].shape[1],\
        ("Image dimensions don't match ground truth")

    X, y = [], []
    for i in range(n):
        img_p = image_patches(images[i], h, w)
        gt_p = image_patches(gts[i], h, w)
        X += img_p
        y += [patch_to_label(patch) for patch in gt_p]

    X = np.asarray(X)
    y = np.asarray(y)

    c1 = sum(y)     # roads
    c0 = len(y)-c1  # background

    print("\tNumber of data points per class: background=%s road=%s" % (c0, c1))

    print("Data loaded.", "X has shape", X.shape, "y has shape", y.shape)
    return X, y

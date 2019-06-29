import numpy as np
import os
from PIL import Image
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt


def test_aug(X, Y, seq):
    imx = seq(images=X)
    imy = seq(images=Y)

    n = min(len(imx), 2)
    fig, ax = plt.subplots(2 * n, 2, figsize=(15, 15))
    for i in range(0, 2 * n, 2):
        j = i // 2
        ax[i, 0].imshow(X[j])
        ax[i, 0].set_title("Orig")
        ax[i, 1].imshow(imx[j])
        ax[i, 1].set_title("Augmented")

        ax[i + 1, 0].imshow(np.argmax(Y[j], axis=-1), cmap=plt.cm.binary_r)
        ax[i + 1, 0].set_title("Orig")
        ax[i + 1, 1].imshow(np.argmax(imy[j], axis=-1), cmap=plt.cm.binary_r)
        ax[i + 1, 1].set_title("Augmented")
    return imx, imy, fig


def one_hot_gt(X):
    dim = X.shape[0]
    label = np.zeros((dim, dim, 2))
    for c in range(2):
        label[:, :, c] = (X == c).astype(int)
    return label


def load_data(one_hot=True, grey=False, as_float=True, resize=True, dim=224):
    print("Loading data...")
    TRAIN_PATH = "data/training/"
    TEST_PATH = "data/test_images/"
    GT_PATH = TRAIN_PATH + "groundtruth/"
    IMG_PATH = TRAIN_PATH + "images/"

    file_names = [f for _, _, f in os.walk(IMG_PATH)][0]
    images, gts = [], []
    print("Loading images and ground truths")
    for f in file_names:
        img = Image.open(IMG_PATH + f)
        gt = Image.open(GT_PATH + f)
        gt = gt.point(lambda p: p > 100 and 255)  # binarize 0/255
        if grey:
            img = img.convert("L")
        if resize:
            img = img.resize((dim, dim))
            gt = gt.resize((dim, dim))
        img = np.array(img)
        if grey:
            # img = img[:, :, np.newaxis]
            img = np.stack([img] * 3, axis=-1)
        gt = np.array(gt)
        if as_float:
            img = img.astype("float32")
            gt = gt.astype("float32")
            img /= 255.0
            gt /= 255.0
        if one_hot:
            gt = one_hot_gt(gt)
        images.append(img)
        gts.append(gt)

        X = np.asarray(images)
        y = np.asarray(gts)
    print("Data ready!\n")
    return X, y, file_names


def load_test(as_float=True, grey=False, resize=True, dim=224):
    TEST_PATH = "data/test_images/"

    file_names = [f for _, _, f in os.walk(TEST_PATH)][0]
    images, gts = [], []
    print("Loading test images")
    for f in file_names:
        img = Image.open(TEST_PATH + f)
        if grey:
            img = img.convert("L")
        if resize:
            img = img.resize((dim, dim))
        img = np.array(img)
        if grey:
            # img = img[:, :, np.newaxis]
            img = np.stack([img] * 3, axis=-1)
        if as_float:
            img = img.astype("float32")
            img /= 255.0

        images.append(img)

        X = np.asarray(images)
    print("Test data ready!\n")
    return X, file_names

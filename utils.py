import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scripts.mask_to_submission import *

TRAIN_PATH = "data/training/"
TEST_PATH = "data/test_images/"
GT_PATH = TRAIN_PATH + "groundtruth/"
IMG_PATH = TRAIN_PATH + "images/"


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

    print("\tData loaded.", "X has shape", X.shape, "y has shape", y.shape)
    return X, y

import os
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn import metrics
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
                 runs_dir="runs",
                 patience=0,
                 patch_width=16,
                 patch_height=16,
                 use_class_weights=True,
                 pred_dir="predictions"):
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patience = patience
        self.use_class_weights = use_class_weights
        self.runs_dir = runs_dir
        self.pred_dir = pred_dir


def img2rgb(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def vis_pred(X, X_preds, n=5, last_n=False, img_sz=400, sz=16):
    image_orig = patches2images(X, img_sz)
    for show in range(n):
        fac = -1 if last_n else 1
        fig, ax = plt.subplots(1, 2, figsize=(5, 5))
        ax[0].imshow(img2rgb(image_orig[show*fac]))
        ax[0].set_title("Orig image")
        ax[1].imshow(X_preds[show*fac], cmap=plt.cm.binary_r)
        ax[1].set_title("Prediction")


def reconstruct_gt(idx, sz=16, plot=False):
    img_name = "satImage_%.3d.png" % idx
    fn = GT_PATH + img_name
    assert os.path.isfile(fn), \
        "Image file %s doesn't exist" % fn

    orig_img = mpimg.imread(fn)
    img_sz = orig_img.shape[0]

    patches = __get_patches(orig_img)
    labels = [patch_to_label(patch) for patch in patches]
    labels = np.asarray(labels)
    rec_img = patches2images(labels, img_sz, sz)

    if plot:
        fig, (orig_ax, rec_ax) = plt.subplots(1, 2, figsize=(10, 10))
        orig_ax.set_axis_off()
        orig_ax.set_title('Original Ground Truth')
        orig_ax.imshow(orig_img, cmap=plt.cm.binary_r)

        rec_ax.set_axis_off()
        rec_ax.set_title('Reconstructed Ground Truth')
        rec_ax.imshow(rec_img[0], cmap=plt.cm.binary_r)
        plt.show()

    return orig_img, rec_img[0]


def images2patches(images, sz=16, labels=False):
    img_sz = images[0].shape[0]
    patches_per_image = (img_sz//sz)**2
    print("\tImage dimensions WxH: %dx%d" % (img_sz, img_sz))
    print("\t%d patches of size %dx%d per image" %
          (patches_per_image, sz, sz))

    n = len(images)
    X = []
    for i in range(n):
        img_p = __get_patches(images[i], sz)
        if labels:
            X += [patch_to_label(patch) for patch in img_p]
        else:
            X += img_p
    X = np.asarray(X)
    return X


def __get_patches(img, sz=16):
    list_patches = []
    img_height = img.shape[0]
    img_width = img.shape[1]
    for i in range(0, img_height, sz):
        for j in range(0, img_width, sz):
            if len(img.shape) < 3:
                im_patch = img[i:i+sz, j:j+sz]
            else:
                im_patch = img[i:i+sz, j:j+sz, :]
            list_patches.append(im_patch)
    return list_patches


# def image_from_labels(labels, img_height, img_width, w=16, h=16):
#     # not needed anymore.. patches2images accomplishes the same
#     img = np.zeros([img_width, img_height])
#     idx = 0
#     for i in range(0, img_height, h):
#         for j in range(0, img_width, w):
#             if labels[idx] > 0.5:  # road
#                 img[i:i+h, j:j+w] = 1
#             else:
#                 img[i:i+h, j:j+w] = 0
#             idx = idx + 1
#     return img


def patches2images(patches, img_sz=400, sz=16):
    ppi = (img_sz//sz)**2
    total_images = patches.shape[0]//ppi
    rec = []
    for i in range(total_images):
        img = __get_image(patches[i*ppi:i*ppi+ppi], img_sz, sz)
        rec.append(img)
    return np.asarray(rec)


def __get_image(patches, img_sz=400, sz=16):
    cols = img_sz//sz
    is_gt = len(patches[0].shape) < 3
    if is_gt:
        rec = np.zeros((img_sz, img_sz))
    else:
        rec = np.zeros((img_sz, img_sz, 3))
    for i in range(0, cols):
        for j in range(0, cols):
            ii = i*sz
            jj = j*sz
            if is_gt:
                rec[ii:ii+sz, jj:jj+sz] = patches[i*cols+j]
            else:
                rec[ii:ii+sz, jj:jj+sz, :] = patches[i*cols+j]
    return rec


def load_tests(sz=16, patches=True):
    print("Loading data...")
    fn = [f for _, _, f in os.walk(TEST_PATH)][0]

    print("\tLoading test images")
    images = []
    for f in fn:
        img = mpimg.imread(TEST_PATH + f)
        images.append(img)

    n = len(images)
    print("\tLoaded", n, "test images")

    img_sz = images[0].shape[0]
    patches_per_image = (img_sz//sz)**2
    print("\tImage dimensions WxH: %dx%d" % (img_sz, img_sz))
    print("\t%d patches of size %dx%d per image" % (patches_per_image, sz, sz))

    if patches:
        X = images2patches(images, sz)
    else:
        X = np.asarray(images)

    print("Data loaded.", "X has shape", X.shape)
    return X, fn


def load_data(sz=16, patches=True):
    print("Loading data...")
    files = [f for _, _, f in os.walk(IMG_PATH)][0]

    print("\tLoading images and ground truths")
    images, gts = [], []
    for f in files:
        img = mpimg.imread(IMG_PATH + f)
        gt = mpimg.imread(GT_PATH + f)
        images.append(img)
        gts.append(gt)

    n = len(images)
    print("\tLoaded", n, "images and", len(gts), "ground truths")
    assert n == len(gts), "no. of images not equal to ground truths"

    if patches:
        X = images2patches(images, sz)
        y = images2patches(gts, sz, True)
        c1 = y.sum()     # roads
        c0 = y.shape[0]-c1  # background

        print("\tNumber of images per class: background=%s road=%s" % (c0, c1))
    else:
        X = np.asarray(images)
        y = np.asarray(gts)

    print("Data loaded.", "X has shape", X.shape, "y has shape", y.shape)
    return X, y, files


########################################################################
# COPIED FROM mask_to_submission.py with slight modificatoin to regexp #
########################################################################
# percentage of pixels > 1 required to assign a foreground label to a patch
foreground_threshold = 0.25

# assign a label to a patch


def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"(\d+).png", image_filename).group(1))
    im = mpimg.imread(image_filename)
    im = im.astype(np.int)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('Id,Prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s)
                         for s in mask_to_submission_strings(fn))

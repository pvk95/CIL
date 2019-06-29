import seed
from model import FCN8
from utils import load_data, test_aug, load_test

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import gc
import imgaug.augmenters as iaa
import glob

# 608px: FCN8_20190629-131128
# 400px: FCN8_20190629-115026

# X -= 0.5
# X -= X.mean()  # Global Centering
# X -= X.mean(axis=0)  # Global Centering (featurewise)
# X -= np.mean(X, axis=(1, 2), keepdims=True)  # local centering(samplewise)


# fcn = FCN8(epochs=30, restore="FCN8_20190628-035713", baseline=0.81)
# learning rate: low
from tensorflow._api.v1.keras import optimizers

X, Y, files = load_data(resize=True, dim=608, grey=False)
print("dim:", X.shape[1], X.shape[2])

# step 1
opt = optimizers.Adam(lr=0.0001)
fcn = FCN8(
    batch_size=1, dropout=0.5, optimizer=opt, patience=15, epochs=100, tune_level=5
)
fcn.model.summary()
fcn.train(X, Y, test_size=0.1)
print(os.path.basename(os.path.normpath(fcn.model_dir)))


# step 2
# chkpt = "FCN8_20190629-070628"
# opt = optimizers.Adam(lr=0.0001)
# fcn = FCN8(
#     dropout=0.5, optimizer=opt, patience=15, epochs=100, restore=chkpt, tune_level=2
# )
# fcn.model.summary()
# fcn.train(X, Y, test_size=0.1)
# print(os.path.basename(os.path.normpath(fcn.model_dir)))


# # step 3
# chkpt = "FCN8_20190629-071515"
# opt = optimizers.Adam(lr=0.00001)
# fcn = FCN8(
#     dropout=0.5, optimizer=opt, patience=15, epochs=100, restore=chkpt, tune_level=4
# )
# fcn.model.summary()
# fcn.train(X, Y, test_size=0.1)
# print(os.path.basename(os.path.normpath(fcn.model_dir)))

# # step 4
# chkpt = "FCN8_20190629-073309"
# opt = optimizers.Adam(lr=0.00001)
# fcn = FCN8(
#     dropout=0.5, optimizer=opt, patience=15, epochs=100, restore=chkpt, tune_level=5
# )
# fcn.model.summary()
# fcn.train(X, Y, test_size=0.1)
# print(os.path.basename(os.path.normpath(fcn.model_dir)))

# step 5
# chkpt = "FCN8_20190629-074544"
# opt = optimizers.Adam(lr=0.0001)
# fcn = FCN8(
#     dropout=0.5, optimizer=opt, patience=15, epochs=100, restore=chkpt, tune_level=5
# )
# fcn.model.summary()
# fcn.train(X, Y, test_size=0.1)
# print(os.path.basename(os.path.normpath(fcn.model_dir)))
# # FCN8_20190629-075549


# opt = optimizers.Adam(lr=0.001)
# fcn = FCN8(dropout=0.5, optimizer=opt, patience=15, epochs=100, restore=chkpt)
# fcn.model.summary()
# fcn.train(X, Y, test_size=0.7)


# opt1 = optimizers.Adam(lr=0.00005)
# fcn.fine_tune(5, optimizer=opt1)
# fcn.fine_tune(4, optimizer="adam")
# fcn.fine_tune(5, lr=1e-4)


# idx = 4
# pred = fcn.model.predict(X[idx:idx+1])
# pred = np.argmax(pred, axis=-1)
# # pred
# plt.imshow(pred[0], cmap=plt.cm.gray)
# # gt
# plt.imshow(np.argmax(Y[idx], axis=-1), cmap=plt.cm.gray)
# # orig
# plt.imshow(X[idx])


# # resize prediction
# im = Image.fromarray(pred[0].astype(np.float))
# res_im = im.resize((400, 400))
# plt.imshow(np.array(res_im))
# loss: 0.3637 - acc: 0.8310 - iou: 0.4767 - val_loss: 0.3408 - val_acc: 0.8493 - val_iou: 0.5200

# from imgaug import augmenters as iaa

# affine = iaa.Affine(
#     # rotate=(-180, 180),
#     # # shear=(-5, 5),
#     # scale=(0.9, 1.1),
#     # mode=["reflect"],
#     # rotate=(-45, 45),
#     # shear=(-16, 16),
# )


# seq = iaa.Sequential(
#     [
#         iaa.Crop(
#             px=(0, 100)
#         ),  # crop images from each side by 0 to 16px (randomly chosen)
#         # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
#         # iaa.GaussianBlur(sigma=(0, 1)),  # blur images with a sigma of 0 to 3.0
#         # iaa.Sometimes(1, iaa.ElasticTransformation(alpha=(3.5, 3.5), sigma=3.25))
#         # iaa.ContrastNormalization((0.5, 1.0), per_channel=0.5),
#     ]
# ).to_deterministic()

# imx, imy, fig = test_aug(X[50:53], Y[50:53], seq)

# # seq_both = iaa.Sequential(
# #     [iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Sometimes(0.5, affine)], random_order=False
# # ).to_deterministic()


# x = img_float_to_uint8(X[1]
# y = np.expand_dims(np.argmax(Y[0], axis=-1), axis=2)
# imx = Image.fromarray(x)
# imy = Image.fromarray((y * 255)[:, :, 0].astype(np.uint8))


# def img_float_to_uint8(img):
#     rimg = img - np.min(img)
#     rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
#     return rimg


# # Submission Area

# fcn600 = FCN8(restore="FCN8_20190629-131128")
# fcn400 = FCN8(restore="FCN8_20190629-115026")

# X_test, testfiles = load_test(resize=True, dim=400,grey=True)
# Y_pred = fcn400.model.predict(X_test, batch_size=1)
# Y_pred2 = np.argmax(Y_pred, axis=-1)

# ############
# # Save predictions
# pred_dir = "predsFourpx"
# if not os.path.isdir(pred_dir):
#     os.mkdir(pred_dir)

# for i in range(0, Y_pred2.shape[0]):
#     file = os.path.join(pred_dir, testfiles[i])
#     # convert to rgb
#     rgb = Y_pred2[i] - np.min(Y_pred2[i])
#     rgb = (rgb / np.max(rgb) * 255).round().astype(np.uint8)
#     im = Image.fromarray(rgb)
#     im = im.resize((608, 608))
#     im.save(file)


# dir_path = os.path.join(pred_dir)
# pattern = os.path.join(dir_path, "test_*.png")
# submission_filename = os.path.join(dir_path, "submission600.csv")
# image_filenames = glob.glob(pattern)
# for image_filename in image_filenames:
#     print(image_filename)

# masks_to_submission(submission_filename, *image_filenames)


# import re
# import matplotlib.image as mpimg

# foreground_threshold = 0.25


# def patch_to_label(patch):
#     df = np.mean(patch)
#     if df > foreground_threshold:
#         return 1
#     else:
#         return 0


# def mask_to_submission_strings(image_filename):
#     """Reads a single image and outputs the strings that should go into the submission file"""
#     img_number = int(re.search(r"\d+", image_filename).group(0))
#     im = mpimg.imread(image_filename)
#     im = im.astype(np.int)
#     patch_size = 16
#     for j in range(0, im.shape[1], patch_size):
#         for i in range(0, im.shape[0], patch_size):
#             patch = im[i : i + patch_size, j : j + patch_size]
#             label = patch_to_label(patch)
#             yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


# def masks_to_submission(submission_filename, *image_filenames):
#     """Converts images into a submission file"""
#     with open(submission_filename, "w") as f:
#         f.write("Id,Prediction\n")
#         for fn in image_filenames[0:]:
#             f.writelines("{}\n".format(s) for s in mask_to_submission_strings(fn))


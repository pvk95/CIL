import seed
from model import FCN8
from utils import load_data

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import gc


# X -= 0.5
# X -= X.mean()  # Global Centering
# X -= X.mean(axis=0)  # Global Centering (featurewise)
# X -= np.mean(X, axis=(1, 2), keepdims=True)  # local centering(samplewise)


# fcn = FCN8(epochs=30, restore="FCN8_20190628-035713", baseline=0.81)
# learning rate: low
from tensorflow._api.v1.keras import optimizers

X, Y, files = load_data(resize=False, dim=224)
print("dim:", X.shape[1], X.shape[2])

# step 1
opt = optimizers.Adam(lr=0.0001)
fcn = FCN8(dropout=0.5, optimizer=opt, patience=15, epochs=100, tune_level=5)
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
#     rotate=(-180, 180),
#     # shear=(-5, 5),
#     scale=(0.9, 1.1),
#     mode=["reflect"],
# )

# seq_both = iaa.Sequential(
#     [iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Sometimes(0.5, affine)], random_order=False
# ).to_deterministic()

# imx, imy, fig = test_aug(X[50:53], Y[50:53], seq_both)

# # x = img_float_to_uint8(X[1]
# # y = np.expand_dims(np.argmax(Y[0], axis=-1), axis=2)
# # imx = Image.fromarray(x)
# # imy = Image.fromarray((y * 255)[:, :, 0].astype(np.uint8))


# def img_float_to_uint8(img):
#     rimg = img - np.min(img)
#     rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
#     return rimg


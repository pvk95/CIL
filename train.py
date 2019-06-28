import seed
from model import FCN8
from utils import load_data

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


X, Y, files = load_data()

# X -= 0.5
# X -= X.mean()  # Global Centering
# X -= X.mean(axis=0)  # Global Centering (featurewise)
# X -= np.mean(X, axis=(1, 2), keepdims=True)  # local centering(samplewise)


# fcn = FCN8(epochs=30, restore="FCN8_20190628-035713", baseline=0.81)
fcn = FCN8(dropout=0.5, epochs=30)
fcn.model.summary()
fcn.train(X, Y)

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

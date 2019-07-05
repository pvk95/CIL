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
import re
import matplotlib.image as mpimg

foreground_threshold = 0.25


def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    im = im.astype(np.int)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i : i + patch_size, j : j + patch_size]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, "w") as f:
        f.write("Id,Prediction\n")
        for fn in image_filenames[0:]:
            f.writelines("{}\n".format(s) for s in mask_to_submission_strings(fn))


from tensorflow._api.v1.keras import optimizers

# load training data
X, Y, files = load_data(resize=True, dim=608, grey=False)
print("dim:", X.shape[1], X.shape[2])

# Build model and train
opt = optimizers.Adam(lr=0.0001)
fcn = FCN8(
    batch_size=1, dropout=0.5, optimizer=opt, patience=11, epochs=1000, tune_level=5
)
fcn.model.summary()
fcn.train(X, Y, test_size=0.1)
chkpt_dir = os.path.basename(os.path.normpath(fcn.model_dir))

# Restore model from the best checkpoint and predict on test images
fcn600 = FCN8(restore=chkpt_dir)

X_test, testfiles = load_test(resize=False)
Y_pred = fcn600.model.predict(X_test, batch_size=1)
Y_pred2 = np.argmax(Y_pred, axis=-1)

############
# Save predictions
pred_dir = "predictions"
if not os.path.isdir(pred_dir):
    os.mkdir(pred_dir)

for i in range(0, Y_pred2.shape[0]):
    file = os.path.join(pred_dir, testfiles[i])
    # convert to rgb
    rgb = Y_pred2[i] - np.min(Y_pred2[i])
    rgb = (rgb / np.max(rgb) * 255).round().astype(np.uint8)
    im = Image.fromarray(rgb)
    # im = im.resize((608, 608))
    im.save(file)


dir_path = os.path.join(pred_dir)
pattern = os.path.join(dir_path, "test_*.png")
submission_filename = os.path.join(dir_path, "submission600.csv")
image_filenames = glob.glob(pattern)
for image_filename in image_filenames:
    print(image_filename)

masks_to_submission(submission_filename, *image_filenames)


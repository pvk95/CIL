#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re
import sys
import glob

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
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    im = (im[:, :, 0]).astype(np.int)
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


if __name__ == '__main__':
    save_folder = sys.argv[1]
    submission_filename = save_folder + sys.argv[2]
    image_filenames = glob.glob(save_folder + 'pred_imgs/test_img_*.png')
    for image_filename in image_filenames:
        print(image_filename)
    masks_to_submission(submission_filename, *image_filenames)

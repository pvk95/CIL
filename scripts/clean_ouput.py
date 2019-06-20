import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import skimage.morphology
import matplotlib.cm as cm
import scipy
import sys
import os

'''
This file takes two inputs:
    - the Folder which has all the pred_imgs stored
    - the Folder the cleaned images are written to

for example: python3 script/clean_output.py Combined/ Cleaned/
This will look into the Combined folder for predicted images, clean them up and write them into the "Cleaned"-folder


This small script applies a morphological opening on the already created output files
and thus cleans them from any "salt"-noise (random white pixels)
This script should be applied before running mask_to_submission.py
'''

if __name__ == "__main__":
    folder = sys.argv[1]
    new_folder = sys.argv[2]

    # iterate over all images
    dir_images = os.path.join(folder, 'pred_imgs')
    for image_name in os.listdir(dir_images):

        # open the image
        image_path = os.path.join(dir_images, image_name)
        image = mpimage.imread(image_path)

        # apply morphological opneing
        image = image[:,:,0]
        opened_image = skimage.morphology.opening(image, np.ones((4,4)))

        # create folder if not exists
        new_path = os.path.join(new_folder, 'pred_imgs')
        if not os.path.exists(new_folder):
            os.makedirs(new_path)

        # save image
        new_path = os.path.join(new_path, image_name)
        plt.imsave(new_path, opened_image, cmap=cm.gray)
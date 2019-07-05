import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import skimage.morphology
import matplotlib.cm as cm
import scipy
import sys
import os

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
        image = (image == False).astype(np.int64)

        # create folder if not exists
        new_path = os.path.join(new_folder, 'pred_imgs')
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        # save image
        new_path = os.path.join(new_path, image_name)
        plt.imsave(new_path, image, cmap=cm.gray)
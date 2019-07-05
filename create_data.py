import sklearn.model_selection
import random
import os
import matplotlib.pyplot as plt
import numpy as np

def getData():
    images = []
    gt = []
    for i in range(1, 101):
        if (i <= 9):
            # fileName = 'cells/00{}cell.png'.format(i)
            fileName_img = 'data/training/images/satImage_00{}.png'.format(i)
            fileName_gt = 'data/training/groundtruth/satImage_00{}.png'.format(i)
        elif (i < 100):
            # fileName = 'cells/0{}cell.png'.format(i)
            fileName_img = 'data/training/images/satImage_0{}.png'.format(i)
            fileName_gt = 'data/training/groundtruth/satImage_0{}.png'.format(i)
        else:
            # fileName = 'cells/{}cell.png'.format(i)
            fileName_img = 'data/training/images/satImage_{}.png'.format(i)
            fileName_gt = 'data/training/groundtruth/satImage_{}.png'.format(i)
        im = plt.imread(fileName_img)
        im_gt = plt.imread(fileName_gt)
        # im = plt.imread(fileName)
        images.append(im)
        gt.append(im_gt)

    images = np.stack(images, axis=0)
    gt = np.stack(gt, axis=0)

    gt = gt[:,:,:,None]

    return [images, gt]

def sampleImage(image, output_shape, start_point):
    x_end = start_point[0] + output_shape[0]
    y_end = start_point[1] + output_shape[1]

    sample = image[start_point[0]: x_end, start_point[1]:y_end,:]
    return sample

def getNumSamples(image1, image2, num, sample_shape):
    assert(image1.shape[:2] == image2.shape[:2])
    x_range = image1.shape[0] - sample_shape[0]
    y_range = image1.shape[1] - sample_shape[1]

    samples1 = []
    samples2 = []

    for _ in range(num):
        x_start = random.randint(0, x_range)
        y_start = random.randint(0, y_range)
        starting_point = (x_start, y_start)

        sample1 = sampleImage(image1, sample_shape, starting_point)
        sample2 = sampleImage(image2, sample_shape, starting_point)

        samples1.append(sample1)
        samples2.append(sample2)
    
    return samples1, samples2


def getSamples(images1, images2, num_samples, sample_shape):
    samples1 = []
    samples2 = []

    for image1, image2 in zip(images1, images2):
        s1, s2 = getNumSamples(image1, image2, num_samples, sample_shape)
        samples1.append(s1)
        samples2.append(s2)

    samples1 = np.vstack(samples1)
    samples2 = np.vstack(samples2)

    return samples1, samples2

def createDataset(save_path, sample_shape, num_samples):
    images, gt = getData()
    image_train, image_test, gt_train, gt_test = sklearn.model_selection.train_test_split(images, gt)

    image_subsample_train, gt_subsample_train = getSamples(image_train, gt_train, num_samples, sample_shape)
    image_subsample_test, gt_subsample_test = getSamples(image_test, gt_test, num_samples, sample_shape)


    train_image_path = os.path.join(save_path, 'train_image.npy')
    test_image_path = os.path.join(save_path, 'test_image.npy')
    train_gt_path = os.path.join(save_path, 'train_gt.npy')
    test_gt_path = os.path.join(save_path, 'test_gt.npy')

    np.save(train_image_path, image_subsample_train)
    np.save(test_image_path, image_subsample_test)
    np.save(train_gt_path, gt_subsample_train)
    np.save(test_gt_path, gt_subsample_test)

if __name__ == "__main__":
    createDataset('data/', (64, 64), 20)
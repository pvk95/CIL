import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def patch2img(X,npc_img = 4, stride=200):

    n_patches = X.shape[0]
    n_samples = n_patches//npc_img
    n_strides = int(npc_img**1/2)

    X_tr = np.zeros((n_samples,400,400,3))
    
    count = 0
    for samp in range(n_samples):
        for j in range(n_strides):
            for i in range(n_strides):
                X_tr[samp,j*stride:j*stride+stride,i*stride:i*stride+stride,:] = X[count,...]
                count = count + 1
    return X_tr

def getPatches(X,stride=200):
    #X is a square image
    
    n_samples = X.shape[0]
    im_sz = X.shape[1]
    n_strides = im_sz//stride
    X_patches = []
    
    for samp in range(n_samples):
        for j in range(n_strides):
            for i in range(n_strides):
                im_patch = X[samp,j*stride:j*stride+stride,i*stride:i*stride+stride,:]                
                X_patches.append(im_patch)

    X_patches = np.stack(X_patches,axis=0)

    return X_patches

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

    gt = gt[:, :, :, None]
    gt = (gt >= 0.5).astype(np.int)

    test_imgs_file = np.array(glob.glob('data/test_images/test_*.png'))
    idxs = np.array([int(x.split('.')[0].split('_')[-1]) for x in test_imgs_file])
    idxs_sort = np.argsort(idxs)
    idxs = idxs[idxs_sort]
    test_imgs_file = test_imgs_file[idxs_sort]
    test_imgs = []
    for img_file in test_imgs_file:
        test_imgs.append(plt.imread(img_file))

    test_imgs = np.stack(test_imgs, axis=0)

    return [images, gt, test_imgs, idxs]


def genRandom(img, segs):
    img_modify = img.copy()
    segs_modify = segs.copy()
    rot = np.random.randint(1, 5, size=1)[0]
    # print(rot)
    flip = np.random.randint(0, 2, size=1)[0]
    # print(flip)
    if (flip):
        img_modify = np.flipud(img_modify)
        segs_modify = np.flipud(segs_modify)

    img_modify = np.rot90(img_modify, rot)
    segs_modify = np.rot90(segs_modify, rot)
    if (flip != 0 or rot < 4):
        # print("Generated!")
        return [img_modify, segs_modify, flip, rot]
    else:
        # print("Aborted!")
        return [None, None, None, None]


def data_augment(X, y, total_samples=200):
    # This function augments the data to make it upto total_samples
    # Set the seed for consistency

    np.random.seed(0)
    n_samples = X.shape[0]
    diff_len = total_samples - n_samples

    X_augment = []
    y_augment = []
    flip_augment = []
    rot_augment = []
    idxs_augment = []
    sel = np.zeros(n_samples)
    while True:
        idx = np.random.choice(range(n_samples))
        if (sel[idx] == 1):
            continue
        X_modify = X[idx, :, :, :]
        y_modify = y[idx, :, :, :]
        X_modify, segs_modify, flip, rot = genRandom(img=X_modify, segs=y_modify)

        # Check for previous
        prev_idx = np.where(idxs_augment == idx)[0]
        if (len(prev_idx) > 0):
            prev_idx = prev_idx[0]
            if (flip == flip_augment[prev_idx] and rot == rot_augment[prev_idx]):
                continue
        if (X_modify is not None):
            sel[idx] = 1
            X_augment.append(X_modify)
            y_augment.append(segs_modify)
            flip_augment.append(flip)
            rot_augment.append(rot)
            idxs_augment.append(idx)
        if (len(X_augment) == diff_len):
            break
        if (np.all(sel == 1)):
            sel = np.zeros(n_samples)

    X_augment = np.stack(X_augment)
    y_augment = np.stack(y_augment)
    flip_augment = np.array(flip_augment)
    rot_augment = np.array(rot_augment)
    idxs_augment = np.array(idxs_augment)
    flip_rot = np.stack((idxs_augment, flip_augment, rot_augment), axis=-1)
    X = np.concatenate((X, X_augment), axis=0)
    y = np.concatenate((y, y_augment), axis=0)

    return [X, y, flip_rot]

def getPredImgs(y_pred,file_names,save_folder):    
    if len(y_pred.shape)==4:
        y_pred = y_pred[:,:,:,0]

    print(y_pred.shape)
    if not os.path.exists(save_folder + 'pred_imgs/'):
        os.makedirs(save_folder + 'pred_imgs/')

    for i in range(len(file_names)):
        fileName = save_folder + 'pred_imgs/test_img_{}.png'.format(file_names[i])
        plt.imsave(fileName,y_pred[i,:,:],cmap = cm.gray)

def resize_to_tr(X):
    n_samples = X.shape[0]
    width = 400
    height = 400
    dim = (width, height)
    resize_imgs = []
    for i in range(n_samples):
        resize_imgs.append(cv2.resize(X[i,...],dim,interpolation = cv2.INTER_CUBIC))
    resize_imgs = np.stack(resize_imgs, axis=0)
    return resize_imgs


def resize_to_test(X):
    n_samples = X.shape[0]
    width = 608
    height = 608
    dim = (width, height)
    resize_imgs = []
    for i in range(n_samples):
        temp = cv2.resize(X[i,...].astype(np.float), dim, interpolation=cv2.INTER_CUBIC)
        temp = (temp>=0.5).astype(np.int)
        resize_imgs.append(temp)

    resize_imgs = np.stack(resize_imgs, axis=0)
    return resize_imgs

def produce_patch(X_test):
    patches = []
    for image in X_test:
        for x in range(0, 209, 52):
            for y in range(0, 209, 52):
                patch = image[x:x+400, y:y+400]
                patches.append(patch)
    return patches

def reconstruct_patches(X_patches):
    chunks = np.split(X_patches, X_patches.shape[0]//25)
    images = []
    for chunk in chunks:
        aggregate_image = np.empty((608,608,25))
        aggregate_image[:,:,:] = np.nan
        counter = 0
        for x in range(0, 209, 52):
            for y in range(0, 209, 52):
                aggregate_image[x: x+400, y:y+400,counter] = chunk[counter,:,:,0]
                counter +=1
        image = np.nanmedian(aggregate_image, axis=-1) 
        image = (image >=0.5).astype(np.int)
        images.append(image)
    return images



if __name__ == '__main__':
    X, y, X_test, file_names = getData()
    xr = resize_to_tr(X_test)
    yr = resize_to_test(y[:, :, :, 0])
    getPredImgs(X_test, file_names, './')

#SegNet architecture: https://arxiv.org/pdf/1511.00561.pdf
import model
import unet
import res_unet
import numpy as np
import argparse
import pickle
import sys
import os
import matplotlib.pyplot as plt
import h5py
import math
import random


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
    gt = (gt>=0.5).astype(np.int)

    return [images, gt]

def genRandom(img, segs):
    img_modify = img.copy()
    segs_modify = segs.copy()
    rot = np.random.randint(1, 5, size=1)[0]
    #print(rot)
    flip = np.random.randint(0, 2, size=1)[0]
    #print(flip)
    if (flip):
        img_modify = np.flipud(img_modify)
        segs_modify = np.flipud(segs_modify)

    img_modify = np.rot90(img_modify, rot)
    segs_modify = np.rot90(segs_modify, rot)
    if (flip != 0 or rot < 4):
        #print("Generated!")
        return [img_modify, segs_modify, flip,rot]
    else:
        #print("Aborted!")
        return [None,None,None,None]

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Road Segmentation')
    parser.add_argument('-batch_sz',default=32,type=int,help = 'Batch Size')
    parser.add_argument('-epochs',default=100,type=int,help = 'No. of epochs')
    parser.add_argument('-lr',default=0.001,type=float,help = 'Learning rate')
    parser.add_argument('-mode',default = 1,type=int,help = 'Training or testing')
    parser.add_argument('-save_folder',default='SegNet/',help='Where to save model')
    parser.add_argument('-frac_train', default=0.95, help='Fraction for training data')
    parser.add_argument('-gpu', default=0, help='GPU number',type=int)
    parser.add_argument('-sz_tr', default=200, help='GPU number', type=int)
    # mode = 1 train and test
    # mode = 2 only train
    # mode = 3 only test
    args = parser.parse_args()

    np.random.seed(0)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    #im_sz = 32 # Square images
    #n_samples = 100
    #n_channels = 3
    #n_outputs = 1
    #lr = 0.001
    #n_epochs = 10

    n_epochs = args.epochs
    lr = args.lr
    batch_sz = args.batch_sz
    mode = args.mode
    save_folder = args.save_folder

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    hyper_param = {}
    for arg in vars(args):
        hyper_param[str(arg)] = getattr(args,arg)

    with open(save_folder + 'hyper_param.txt','w') as f:
        for key in hyper_param.keys():
            f.write(key + ' : ' + str(hyper_param[key]) + '\n')

    if not os.path.isfile('training_data.h5'):
        [X, y] = getData()
        [X, y, flip_rot] = data_augment(X,y,total_samples = args.sz_tr)
        file_data = h5py.File('training_data.h5','w')
        file_data['images'] = X
        file_data['groundTruth'] = y
        file_data['flip_rot'] = flip_rot
        file_data.close()

    else:
        file_data = h5py.File('training_data.h5','r')
        X = file_data['images'][()]
        y = file_data['groundTruth'][()]
        file_data.close()


    im_sz = X.shape[1]  # Square image
    n_samples = X.shape[0]
    n_channels = X.shape[3]

    #Split the trainig and test dataset
    n_train = int(n_samples*args.frac_train)
    idxs_order = np.random.permutation(np.arange(n_samples))
    idxs_train = idxs_order[:n_train]
    idxs_valid = idxs_order[n_train:]

    X_train = X[idxs_train,:,:,:]
    y_train = y[idxs_train,:,:,:]

    X_valid = X[idxs_valid,:,:,:]
    y_valid = y[idxs_valid,:,:,:]

    #Update hyper_param
    hyper_param['idxs_train'] = idxs_train
    hyper_param['idxs_valid'] = idxs_valid

    with open(save_folder + 'hyper_param.pickle','wb') as f:
        pickle.dump(hyper_param,f)

#    model = model.SegNet(im_sz = im_sz,n_channels= n_channels,lr = lr,\
#                          n_epochs=n_epochs,batch_sz=batch_sz,save_folder=save_folder)
    model = res_unet.ResUNet(save_folder,deepness=3, epochs=args.epochs)
    if (mode == 1):
        model.train(X_train,y_train,X_valid,y_valid)
        model.predict(X_valid)
    elif (mode == 2):
        model.train(X_train, y_train, X_valid, y_valid)
    elif (mode ==3):
        model.predict(X_valid)
    else:
        print("Unknown behavior!")
        sys.exit(1)

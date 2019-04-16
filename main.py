#SegNet architecture: https://arxiv.org/pdf/1511.00561.pdf
import model
import unet
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

    return [images, gt]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SegNet architecture')
    parser.add_argument('-batch_sz',default=32,type=int,help = 'Batch Size')
    parser.add_argument('-epochs',default=100,type=int,help = 'No. of epochs')
    parser.add_argument('-lr',default=0.001,type=float,help = 'Learning rate')
    parser.add_argument('-mode',default = 1,type=int,help = 'Training or testing')
    parser.add_argument('-save_folder',default='SegNet/',help='Where to save model')
    # mode = 1 train and test
    # mode = 2 only train
    # mode = 3 only test
    args = parser.parse_args()

    np.random.seed(0)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


    [X,y] = getData()

    im_sz = X.shape[1] # Square image
    n_samples = X.shape[0]
    n_channels = X.shape[3]

    if not os.path.isfile('training_data.h5'):
        file_data = h5py.File('training_data.h5','w')
        file_data['images'] = X
        file_data['groundTruth'] = y
        file_data.close()

    #Split the trainig and test dataset
    n_train = int(n_samples*0.75)
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
    model = unet.UNet(deepness=3, epochs=1000)
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


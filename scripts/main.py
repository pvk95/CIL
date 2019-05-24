#SegNet architecture: https://arxiv.org/pdf/1511.00561.pdf
#Unet architecture:
import models
import utils
import numpy as np
import argparse
import pickle
import sys
import os
import matplotlib.pyplot as plt
import h5py
import math
import random
import glob


def getSegImgs(model, X_test,save_folder):
    X_test = utils.resize_to_tr(X_test)
    y_pred = model.predict(X_test)
    y_pred = utils.resize_to_test(y_pred)
    utils.getPredImgs(y_pred, file_names,save_folder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Road Segmentation')
    parser.add_argument('-batch_sz',default=32,type=int,help = 'Batch Size')
    parser.add_argument('-epochs',default=100,type=int,help = 'No. of epochs')
    parser.add_argument('-lr',default=0.001,type=float,help = 'Learning rate')
    parser.add_argument('-mode',default = 1,type=int,help = 'Training or testing')
    parser.add_argument('-save_folder',default='UNet/',help='Where to save model')
    parser.add_argument('-frac_valid', default=0.1, help='Fraction for training data')
    parser.add_argument('-gpu', default=0, help='GPU number',type=int)
    parser.add_argument('-sz_tr', default=200, help='GPU number', type=int)

    # mode = 1 train,valid and test
    # mode = 2 only train,valid
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
    frac_valid = args.frac_valid
    sz_tr = args.sz_tr

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    hyper_param = {}
    for arg in vars(args):
        hyper_param[str(arg)] = getattr(args,arg)

    with open(save_folder + 'hyper_param.txt','w') as f:
        for key in hyper_param.keys():
            f.write(key + ' : ' + str(hyper_param[key]) + '\n')

    X,y,X_test,file_names = utils.getData()

    im_sz = X.shape[1]  # Square image
    n_samples = X.shape[0]
    n_channels = X.shape[3]

    #Split the trainig and test dataset
    n_valid = int(n_samples*frac_valid)
    idxs_order = np.random.permutation(np.arange(n_samples))
    idxs_valid = idxs_order[:n_valid]
    idxs_train = idxs_order[n_valid:]

    X_train = X[idxs_train,:,:,:]
    y_train = y[idxs_train,:,:,:]

    X_valid = X[idxs_valid,:,:,:]
    y_valid = y[idxs_valid,:,:,:]

    [X_train, y_train, flip_rot] = utils.data_augment(X_train, y_train, total_samples = sz_tr)

    #Update hyper_param
    hyper_param['idxs_train'] = idxs_train
    hyper_param['idxs_valid'] = idxs_valid
    hyper_param['flip_rot'] = flip_rot

    with open(save_folder + 'hyper_param.pickle','wb') as f:
        pickle.dump(hyper_param,f)

#   model = models.SegNet(save_folder=save_folder,epochs = args.epochs)
    model = models.UNet(save_folder,deepness=3, epochs=args.epochs)

    if (mode == 1):
        model.train(X_train,y_train,X_valid,y_valid)
        getSegImgs(model,X_test,save_folder)
    elif (mode == 2):
        model.train(X_train, y_train, X_valid, y_valid)
    elif (mode ==3):
        getSegImgs(model, X_test,save_folder)
    else:
        print("Unknown behavior!")
        sys.exit(1)

'''
if not os.path.isfile('training_data.h5'):
    [X,y,X_test] = getData()
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
'''
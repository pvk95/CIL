import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import pickle
import os
from tensorflow.python import keras
import utils
import argparse

def getPlots(idx,x,y_gt,y_pred,train=False):
    plt.figure(figsize=(10, 5))
    plt.subplot(131)
    plt.imshow(x)
    plt.title('Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(132)
    plt.imshow(y_gt,cmap='gray')
    plt.title('Ground Truth')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(133)
    plt.imshow(y_pred,cmap='gray')
    plt.title('Segmented Image')
    plt.xticks([])
    plt.yticks([])
    if train:
        plt.savefig(save_folder + 'comp_imgs/train_comp_im_{}.png'.format(idx), dpi=200)
    else:
        plt.savefig(save_folder + 'comp_imgs/valid_comp_im_{}.png'.format(idx), dpi=200)

def getComp(idx,x,y):
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(x)
    plt.title('Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(y,cmap='gray')
    plt.title('Segmented')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_folder + 'comp_imgs/test_comp_im_{}.png'.format(idx), dpi=200)

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-save_folder',default='SegNet/',help='Where is the root folder of the experiment?')
    parser.add_argument('-arch',default='segnet',help='Architecture?')

    args = parser.parse_args()
    save_folder = args.save_folder
    arch = args.arch

    #save_folder = 'UNet/'
    #arch = 'unet/'

    X, y, X_test, file_names = utils.getData()
    # X: [100,400,400,3]
    # y: [100,400,400,1]
    # X_test: [94,608,608,3]

    if not os.path.exists(save_folder + 'pred_imgs/'):
        print("No predictions on test data. Exiting ...")
        sys.exit(1)

    if not os.path.exists(save_folder + 'comp_imgs/'):
        os.makedirs(save_folder + 'comp_imgs/')

    for i in range(5):
        seg_im = plt.imread(save_folder + 'pred_imgs/test_img_{}.png'.format(file_names[i]))
        getComp(file_names[i],X_test[i,:,:,:],seg_im)

    os.system('python scripts/mask_to_submission.py {} unet.csv'.format(save_folder))

    valid_file = h5py.File(save_folder + 'validation.h5','r')
    y_valid = valid_file['data']
    # y_valid: [10,400,400]

    with open(save_folder + 'hyper_param.pickle','rb') as f:
        hyper_param = pickle.load(f,encoding="bytes")

    idxs_valid = hyper_param['idxs_valid']

    X_valid = X[idxs_valid,:,:,:]
    y_gt_valid = y[idxs_valid,:,:,:]

    indexes = np.arange(X_valid.shape[0])
    for idx in indexes:
        getPlots(idx,X_valid[idx,...],y_gt_valid[idx,:,:,0],y_valid[idx,:,:,0])

    idxs_train = hyper_param['idxs_train']
    X_train = X[idxs_train,...]
    y_gt_train = y[idxs_train, :, :, :]

    indexes = np.random.choice(np.arange(X_train.shape[0]),size=10,replace=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if arch=='segnet':
        model = keras.models.load_model(save_folder + 'checkpoint/SegNet.h5')
    elif arch=='unet':
        model = keras.models.load_model(save_folder + 'checkpoint/UNet.h5')
    else:
        print("Unknown architecture! Exiting ...")
        sys.exit(1)

    y_pred_train = model.predict(X_train[indexes,...])
    y_pred_train = (y_pred_train>=0.5).astype(np.int)
    
    for i,idx in enumerate(indexes):
        getPlots(idx, X_train[idx,...], y_gt_train[idx,:,:,0], y_pred_train[i,:,:,0],train=True)

    #Training curves
    with open(save_folder + 'train_curves.pickle','rb') as f:
        train_curves = pickle.load(f, encoding = 'bytes')

    training_loss = np.array(train_curves['train'])
    val_loss = np.array(train_curves['val'])

    plt.figure(figsize=(10,5))
    plt.plot(training_loss,label = 'Train')
    plt.plot(val_loss,label = 'Validation')
    plt.title('Training curves')
    plt.legend()
    plt.savefig(save_folder + 'Train_curves.png',dpi = 200)
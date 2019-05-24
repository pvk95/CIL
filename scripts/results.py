import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import pickle
import os
import utils

def getPlots(idx,X_valid,y_valid,y_pred):
    plt.figure(figsize=(10, 5))
    plt.subplot(131)
    plt.imshow(X_valid[idx, :, :, :])
    plt.title('Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(132)
    plt.imshow(y_valid[idx, :, :, 0])
    plt.title('Ground Truth')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(133)
    plt.imshow(y_pred[idx, :, :, 0])
    plt.title('Segmented Image')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_folder + 'im_{}.png'.format(idx), dpi=200)

def getComp(idx,x,y):
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(x)
    plt.title('Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(y)
    plt.title('Segmented')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_folder + 'test_comp_im_{}.png'.format(idx), dpi=200)

if __name__ =='__main__':
    save_folder = sys.argv[1] #Unet/
    #save_folder = 'Unet/'

    X, y, X_test, file_names = utils.getData()

    if not os.path.exists(save_folder + 'pred_imgs/'):
        print("No predictions on test data. Exiting ...")
        sys.exit(1)

    for i in range(5):
        seg_im = plt.imread(save_folder + 'pred_imgs/test_img_{}.png'.format(file_names[i]))
        getComp(file_names[i],X_test[i,:,:,:],seg_im)

    os.system('python scripts/mask_to_submission.py {} unet.csv'.format(save_folder))

    valid_file = h5py.File(save_folder + 'validation.h5','r')
    y_valid = valid_file['data']

    with open(save_folder + 'hyper_param.pickle','rb') as f:
        hyper_param = pickle.load(f,encoding="bytes")

    idxs_valid = hyper_param['idxs_valid']

    X_valid = X[idxs_valid,:,:,:]
    y_gt_valid = y[idxs_valid,:,:,:]

    indexes = np.arange(X_valid.shape[0])
    for idx in indexes:
        getPlots(idx,X_valid,y_gt_valid,y_valid)

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

import matplotlib.image as mpimg

im = mpimg.imread('pred_imgs/test_img_7.png')

np.max(im)

np.min(im)

plt.imshow(im[:,:,1])

im.dtype

plt.hist(im[:,:,0])

np.mean(im[:16,:16])



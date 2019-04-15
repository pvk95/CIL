import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import pickle

if __name__ =='__main__':
    save_folder = sys.argv[1]

    pred_file = h5py.File(save_folder + 'predictions.h5','r')
    y_pred = pred_file['data']

    with open(save_folder + 'hyper_param.pickle','rb') as f:
        hyper_param = pickle.load(f,encoding="bytes")


    print(hyper_param)

    idxs_valid = hyper_param['idxs_valid']

    file_data = h5py.File('training_data.h5','r')
    X = file_data['images'][()]
    y = file_data['groundTruth'][()]

    X_valid = X[idxs_valid,:,:,:]
    y_valid = y[idxs_valid,:,:,:]

    idx = 0
    plt.figure(figsize=(10,5))
    plt.subplot(131)
    plt.imshow(X_valid[idx,:,:,:])
    plt.title('Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(132)
    plt.imshow(y_valid[idx,:,:,0])
    plt.title('Ground Truth')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(133)
    plt.imshow(y_pred[idx,:,:,0])
    plt.title('Segmented Image')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_folder + 'seg.png',dpi = 200)

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


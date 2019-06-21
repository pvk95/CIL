'''
#SegNet architecture: https://arxiv.org/pdf/1511.00561.pdf
#Unet architecture:

Best performing model(SegNet) : python scripts/main.py -lr 0.01 -save_folder SegNet/ -epochs 200 -sz_tr 500 -arch segnet
'''
import sys
sys.path.append('./scripts')
import models
import utils
import numpy as np
import argparse
import os
import pickle
import os
import h5py

def getSegImgs(model, X_test, save_folder):
    X_test = utils.resize_to_tr(X_test)
    if sub_sample:
        X_test = utils.getPatches(X_test)
    y_pred = model.predict(X_test)
    if sub_sample:
        y_pred = utils.patch2img(y_pred)
    
    y_pred = utils.resize_to_test(y_pred)
    utils.getPredImgs(y_pred, file_names, save_folder)

def produceSegmentedImages(model, X_test, save_folder, mode=0):
    if mode == 0:
        # Resize image
        produce_patches = utils.resize_to_tr
        reconstruct_from_patches = utils.resize_to_test
    elif mode == 1:
        # Use patces to recreate image
        produce_patches = utils.produce_patch
        reconstruct_from_patches = utils.reconstruct_patches
    elif mode == 2:
        # pad image to get original
        produce_patches = utils.use_padding
        reconstruct_from_patches = utils.from_padding

    X_patches = produce_patches(X_test)
    X_patches = np.array(X_patches)
    y_patches = model.predict(X_patches)
    y_pred = reconstruct_from_patches(y_patches)
    y_pred = np.array(y_pred)
    utils.getPredImgs(y_pred, file_names, save_folder)
    


def getValid(model, X_valid):
    y_valid = model.predict(X_valid)
    
    if sub_sample:
        y_valid = utils.patch2img(y_valid)
    
    with h5py.File(save_folder + 'validation.h5', 'w') as f:
        f['data'] = y_valid        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Road Segmentation')

    parser.add_argument('-batch_sz',default=4,type=int,help = 'Batch Size')
    parser.add_argument('-epochs',default=100,type=int,help = 'No. of epochs')
    parser.add_argument('-lr',default=0.001,type=float,help = 'Learning rate')
    parser.add_argument('-mode',default = 1,type=int,help = 'Training or testing')
    parser.add_argument('-save_folder',default='UNet/',help='Where to save model')
    parser.add_argument('-frac_valid', default=0.1, help='Fraction for training data')
    parser.add_argument('-gpu', default=0, help='GPU number',type=int)
    parser.add_argument('-sz_tr', default=150, help='No. of samples data aug', type=int)
    parser.add_argument('-arch', default='unet', help='Which architecture? ', type=str)
    parser.add_argument('-sub_sample',default=0,help='Whether to sub sample',type=int)
    parser.add_argument('-rec_mode', default=0, help='What type of image reconstrucitons is used', type=int)
    # -arch == unet or -arch ==segnet

    # mode = 1 train,valid and test
    # mode = 2 only train,valid
    # mode = 3 only test
    args = parser.parse_args()

    np.random.seed(0)

    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # im_sz = 32 # Square images
    # n_samples = 100
    # n_channels = 3
    # n_outputs = 1
    # lr = 0.001
    # n_epochs = 10

    n_epochs = args.epochs
    lr = args.lr
    batch_sz = args.batch_sz
    mode = args.mode
    save_folder = args.save_folder
    frac_valid = args.frac_valid
    sz_tr = args.sz_tr
    arch = args.arch
    sub_sample = not not args.sub_sample
    rec_mode = args.rec_mode

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    hyper_param = {}
    for arg in vars(args):
        hyper_param[str(arg)] = getattr(args, arg)

    with open(save_folder + 'hyper_param.txt', 'w') as f:
        for key in hyper_param.keys():
            f.write(key + ' : ' + str(hyper_param[key]) + '\n')

    #Get the data
    X,y,X_test,file_names = utils.getData()

    n_samples = X.shape[0]
    n_channels = X.shape[3]

    # Split the trainig and test dataset
    n_valid = int(n_samples * frac_valid)
    idxs_order = np.random.permutation(np.arange(n_samples))
    idxs_valid = idxs_order[:n_valid]
    idxs_train = idxs_order[n_valid:]

    X_train = X[idxs_train, :, :, :]
    y_train = y[idxs_train, :, :, :]

    X_valid = X[idxs_valid,:,:,:]
    y_valid = y[idxs_valid,:,:,:]
    
    if sub_sample:
        X_train = utils.getPatches(X_train,stride = 200)
        y_train = utils.getPatches(y_train,stride = 200)
        X_valid = utils.getPatches(X_valid,stride = 200)
        y_valid = utils.getPatches(y_valid,stride = 200)
    
    [X_train, y_train, flip_rot] = utils.data_augment(X_train, y_train, total_samples = sz_tr)
    
    #Update hyper_param
    hyper_param['idxs_train'] = idxs_train
    hyper_param['idxs_valid'] = idxs_valid
    hyper_param['flip_rot'] = flip_rot

    with open(save_folder + 'hyper_param.pickle','wb') as f:
        pickle.dump(hyper_param,f)
    
    input_shape = X_train.shape[1:]
    if(arch == 'segnet'):
        model = models.SegNet(save_folder=save_folder,input_shape= input_shape,epochs = args.epochs)
    elif(arch == 'unet'):
        model = models.UNet(save_folder,input_shape=input_shape,deepness=4, \
                            epochs=args.epochs,batch_size=batch_sz)
    elif(arch == 'resnet'):
        model = models.ResUNet(save_folder=save_folder, epochs = args.epochs)
    elif(arch == 'combined'):
        model = models.CombinedModel(save_folder=save_folder, epochs=args.epochs, batch_size=batch_sz)
    else:
        print("Unknown architecture! Exiting ...")
        sys.exit(1)

    if (mode == 1):
        model.train(X_train,y_train,X_valid,y_valid)
        getValid(model,X_valid)
        produceSegmentedImages(model, X_test, save_folder, rec_mode)
    elif (mode == 2):
        model.train(X_train, y_train, X_valid, y_valid)
        getValid(model,X_valid)
    elif (mode ==3):
        getValid(model,X_valid)
        getSegImgs(model, X_test,save_folder)
    else:
        print("Unknown behavior!")
        sys.exit(1)
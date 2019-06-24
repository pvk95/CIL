import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow._api.v1 import keras
from sklearn.model_selection import train_test_split
from models import *
from utils import *


def normalize_meanstd(X, axis=(1, 2)):
    # sample wise: for each image separately
    print("Standardizing images...")
    mean = np.mean(X, axis=axis, keepdims=True)
    var = ((X - mean)**2).mean(axis=axis, keepdims=True)
    std = np.sqrt(var)
    print("Images standardized")
    return (X - mean) / std


def normalize_meanstd2(X, axis=0):
    # feature wise: across the entire training dataset
    print("Standardizing images...")
    mean = np.mean(X, axis=axis, keepdims=True)
    var = ((X - mean)**2).mean(axis=axis, keepdims=True)
    std = np.sqrt(var)
    print("Images standardized")
    return (X - mean) / std


X, y = load_data(16, 16)
X = normalize_meanstd2(X)
X_train, X_valid, Y_train, Y_valid = train_test_split(X, y,
                                                      test_size=0.1,
                                                      shuffle=True,
                                                      stratify=y)


conf = Config(epochs=1000, patience=30,
              use_class_weights=True, batch_size=10000)
basic_cnn = BasicCNN(config=conf)
basic_cnn.train(X_train, Y_train, X_valid, Y_valid)


# orig, rec = reconstruct_gt(22, PATCH_SIZE, PATCH_SIZE, plot=True)
# plt.imshow(X[22])

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow._api.v1 import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from models import *
from utils import *

X, y = load_data(16, 16)
X_train, X_valid, Y_train, Y_valid = train_test_split(X, y,
                                                      test_size=0.1,
                                                      shuffle=True,
                                                      stratify=y)

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(Y_train),
                                                  Y_train)

conf = Config(epochs=1000, patience=20,
              use_class_weights=True, batch_size=32)
basic_cnn = BasicCNN(config=conf)
basic_cnn.model.summary()
basic_cnn.train(X_train, Y_train, X_valid, Y_valid)
fig = basic_cnn.plots(savefig=True)


# orig, rec = reconstruct_gt(22, PATCH_SIZE, PATCH_SIZE, plot=True)
# plt.imshow(X[22])

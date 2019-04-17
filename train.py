import numpy as np
from unet import UNet

if __name__ == "__main__":
    train_image = np.load('data/train_image.npy')
    test_image = np.load('data/test_image.npy')
    train_gt = np.load('data/train_gt.npy')
    test_gt = np.load('data/test_gt.npy')
    model = UNet((64,64,3), epochs=100, verbose=1, batch_size=64)
    model.train(train_image, train_gt, test_image, test_gt)
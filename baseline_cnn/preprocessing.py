from sklearn.model_selection import train_test_split
import numpy as np


class Preprocessing(object):
    def __init__(self, standardize=True, samplewise=False):
        self.standardize = standardize
        self.mean = None
        self.std = None
        self.samplewise = samplewise

    def fit(self, X):
        if self.standardize and not self.samplewise:
            axis = 0  # samplewise (1, 2)
            print("Collecting stats for standardization...")
            self.mean = np.mean(X, axis=axis, keepdims=True)
            var = ((X - self.mean)**2).mean(axis=axis, keepdims=True)
            self.std = np.sqrt(var)
            print("\tStats computed")

    def transform(self, X, name=""):
        if self.standardize:
            print("Standardizing %s images" % name)
            if self.samplewise:
                axis = (1, 2)
                mean = np.mean(X, axis=axis, keepdims=True)
                var = ((X - mean)**2).mean(axis=axis, keepdims=True)
                std = np.sqrt(var)
                X_norm = (X - mean) / (std+1e-7)
            else:
                X_norm = (X - self.mean) / self.std
                mean = X_norm.mean()
                std = X_norm.std()
                print("\tImages standardized: images_mean=%f images_std=%f" %
                      (mean, std))
            return X_norm

    def fit_transform(self, X, name=""):
        self.fit(X)
        return self.transform(X, name=name)

    def split_data(self, X, y, test_size=0.1, shuffle=True):
        stratify = y if shuffle else None
        if self.standardize:
            X = self.fit_transform(X, name="training")

        X_t, X_v, Y_t, Y_v = train_test_split(X, y,
                                              test_size=test_size,
                                              shuffle=shuffle,
                                              stratify=stratify)
        # if self.standardize:
        #     X_t = self.fit_transform(X_t, name="training")
        #     X_v = self.transform(X_v, name="validation")

        return X_t, X_v, Y_t, Y_v

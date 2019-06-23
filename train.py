from utils import *


PATCH_SIZE = 16


orig, rec = reconstruct_gt(22, PATCH_SIZE, PATCH_SIZE, plot=True)

X, y = load_data(PATCH_SIZE, PATCH_SIZE)

print("X has shape", X.shape)
print("y has shape", y.shape)

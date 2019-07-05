# Import this as the first import in order to reproduce training results
from numpy.random import seed

seed(1234)  # keras

from tensorflow import set_random_seed

set_random_seed(5678)  # TF

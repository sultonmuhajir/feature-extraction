import numpy as np
from skimage.feature import greycomatrix, greycoprops

patch = [
   [0, 0, 1, 1],
   [0, 0, 1, 1],
   [0, 2, 2, 2],
   [2, 2, 3, 3],
]
g = greycomatrix(patch, distances=[1,2,3], angles=[0], levels=4)
np.array(g).flatten().reshape(-1,4,4)
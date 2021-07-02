import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

obj = np.array([
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
[0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.uint8)

plt.imshow(obj, 'gray')
plt.show()



contour = np.array(cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]).reshape(-1, 2)
i = 1
codes = []
while i < contour.shape[0]:
   delta = contour[i] - contour[i - 1]
   codes.append(math.floor((math.atan2(delta[1], delta[0]) / math.pi * 4 + 8) % 8))
   i+=1
print(np.array(codes))
print(np.diff(codes))


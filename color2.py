import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("img/fruit.jpg")
imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([imgRgb], [0, 2], None, [32,32], [0, 256, 0, 256])


plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.subplot(2,2,1), plt.imshow(hist, interpolation = "nearest")
plt.colorbar(plt.imshow(hist, interpolation = "nearest"))


ret,thresh0 = cv2.threshold(imgRgb[:,:,0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh1 = cv2.threshold(imgRgb[:,:,1], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh2 = cv2.threshold(imgRgb[:,:,2], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


plt.subplot(2,2,2), plt.imshow(thresh0, 'gray'), plt.title('Threshold Hue')
plt.subplot(2,2,3), plt.imshow(thresh1, 'gray'), plt.title('Threshold Saturation')
plt.subplot(2,2,4), plt.imshow(thresh2, 'gray'), plt.title('Threshold Value')


plt.show()
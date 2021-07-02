import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("img/fruit.jpg")
imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.subplot(2,2,1), plt.imshow(imgRgb), plt.title('Full Color')
plt.subplot(2,2,2), plt.imshow(imgRgb[:,:,0],'gray'), plt.title('Hue Color')
plt.subplot(2,2,3), plt.imshow(imgRgb[:,:,1],'gray'), plt.title('Saturation Color')
plt.subplot(2,2,4), plt.imshow(imgRgb[:,:,2],'gray'), plt.title('Value Color')


plt.show()
import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("img/fruit.jpg")
imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


plt.subplots_adjust(wspace=0.4, hspace=0.4)
ret,thresh0 = cv2.threshold(imgRgb[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh1 = cv2.threshold(imgRgb[:,:,1],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh2 = cv2.threshold(imgRgb[:,:,2],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


mix = np.multiply(thresh0, thresh1, thresh2)
plt.subplot(2,2,1), plt.imshow(mix, 'gray'), plt.title('Mix')


kernel = np.ones((5, 5), np.uint8)
mix2 = cv2.morphologyEx(mix, cv2.MORPH_OPEN, kernel)
mix2 = cv2.morphologyEx(mix2, cv2.MORPH_CLOSE, kernel)
plt.subplot(2,2,2), plt.imshow(mix2, 'gray'), plt.title('Mix2')


plt.show()
========================== threshold dither ======================


'''Cv2 wins, auto conversion Grayscale'''
'''plt will plot any datatype ... auto detection'''

'''black/white '''
'''vectorized, pretty cool'''

import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('images/ghost.1.0.png',0)
# img = cv2.imread('images/color-image.png',0)
imgMax = np.float64(255)
threshhold = imgMax*0.4   
'''# extreme values to discover eyes! magical'''
img = (img>=threshhold)*1.0
# img = img[10:20,10:20]
# print type(img[0][0])
# print img
imgplot = plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
plt.show()


================================================
======'''# extreme values to discover eyes! magical'''===
================================================
img = cv2.imread('images/stinkbug.png',0)
imgMax = np.float64(255)
for i in range(10):
  threshhold = imgMax*0.1*i
  '''# extreme values to discover eyes! magical'''
  _img = (img>=threshhold)*1.0
  # img = img[10:20,10:20]
  # print type(img[0][0])
  # print img
  imgplot = plt.imshow(_img, cmap = 'gray', interpolation = 'nearest')
  plt.show()
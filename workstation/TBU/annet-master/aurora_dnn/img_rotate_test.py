import cv2
import numpy as np
import random

if __name__=='__main__':
    url = '/home/aurora/hdd/workspace/data/auroua/N20031221G030001.bmp'
    img = cv2.imread(url, 0)
    print img.shape
    rows, cols = img.shape
    cv2.imshow('original', img)
    angle = int(random.random()*360)
    print angle
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow('rotate', dst)
    cv2.waitKey(0)
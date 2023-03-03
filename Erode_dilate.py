import cv2
import numpy as np
from numpy import uint8

kernel = np.ones((3,3), uint8)

img = cv2.imread('data/cam4/mask.png')
img = cv2.dilate(img, kernel)

cv2.imwrite('/Users/macbook/Desktop/mask4_dilate.png', img)
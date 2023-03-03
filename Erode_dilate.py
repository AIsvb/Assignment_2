import cv2
import numpy as np
from numpy import uint8

# kernel = np.ones((3,3), uint8)
#
# img = cv2.imread('data/cam4/mask.png')
# img = cv2.dilate(img, kernel)
#
# cv2.imwrite('/Users/macbook/Desktop/mask4_dilate.png', img)

from BackgroundSubtraction import BackgroundSubtractor as BS

bs1 = BS('data/cam1/background.avi', 'data/cam1/video.avi')
bs2 = BS('data/cam2/background.avi', 'data/cam2/video.avi')
bs3 = BS('data/cam3/background.avi', 'data/cam3/video.avi')
bs4 = BS('data/cam4/background.avi', 'data/cam4/video.avi')

# bs1.subtract_background()
# bs2.subtract_background()
bs3.subtract_background()
# bs4.subtract_background()

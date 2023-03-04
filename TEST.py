import cv2
import numpy as np
import PIL

from LookupTable import LookupTable as LT
from LookupTable import Voxel
#
mask1 = cv2.imread('data/cam1/frames1/21.png')
mask2 = cv2.imread('data/cam1/frames1/22.png')
# mask2 = cv2.imread('data/cam2/mask.png')
mask3 = cv2.imread('data/cam3/mask.png')
mask4 = cv2.imread('data/cam4/mask.png')

# LT = LT(50, 50, 50, mask1, mask2, mask3, mask4)
# LT.create_voxels()
# LT.create_lookup_table()
# list = LT.get_voxels('data/cam1/mask.png', 'data/cam2/mask.png', 'data/cam3/mask.png', 'data/cam4/mask.png')

frame1_a = cv2.imread('/Users/macbook/Desktop/test.png')
frame1_b = cv2.imread('/Users/macbook/Desktop/test2.png')

changed_pixels1 = mask1 ^ mask2

cv2.imwrite('/Users/macbook/Desktop/FILE.png', changed_pixels1)

print("hi")

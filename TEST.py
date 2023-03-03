import cv2
import numpy as np
import PIL

from LookupTable import LookupTable as LT
from LookupTable import Voxel

mask1 = cv2.imread('data/cam1/mask.png')
mask2 = cv2.imread('data/cam2/mask.png')
mask3 = cv2.imread('data/cam3/mask.png')
mask4 = cv2.imread('data/cam4/mask.png')

LT = LT(50, 50, 100, mask1, mask2, mask3, mask4)
LT.create_voxels()
LT.create_dictionary()
list = LT.create_final_list(mask1, mask2, mask3, mask4)

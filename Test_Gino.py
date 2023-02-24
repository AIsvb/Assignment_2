import cv2
from BackgroundSubtraction import BackgroundSubtractor as BS

bs = BS('data/cam3/background.avi', 'data/cam3/video.avi')
bs.subtract_background()


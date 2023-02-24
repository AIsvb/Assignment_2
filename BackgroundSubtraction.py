import cv2 
import numpy as np

class BackgroundSubtractor:
    def __init__(self, background_path, video_path):
        self.background = background_path
        self.video = video_path
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.kernel = np.ones((3,3), np.uint8)

    def subtract_background(self):
        background = cv2.VideoCapture(self.background)
        video = cv2.VideoCapture(self.video)

        while True:
            # Read background video and train background model
            ret, frame = background.read()
            self.fgbg.apply(frame)
            if frame is None:
                break

            # Read video, subtract background without training the background model
            ret, frame = video.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            foreground = self.fgbg.apply(frame, None, 0)
            if frame is None:
                break

            # Postprocessing
            # fgmask = cv2.erode(foreground, self.kernel, iterations=2)
            # fgmask[np.abs(foreground) < 254] = 0
            # fgmask = cv2.dilate(fgmask, self.kernel, iterations=1)
            fgmask = self.find_contours(foreground)

            # Show video
            # cv2.imshow('Frame', frame)
            cv2.imshow('Unprocessed', foreground)
            cv2.imshow('Processed', fgmask)

            keyboard = cv2.waitKey(1)
            if keyboard == 'q' or keyboard == 27:
                break

        video.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(0)

    # Functionality for post-processing: now a duplicate of what already is in subtract_background
    def post_process(self, img):
        # e_shape = cv2.MORPH_RECT
        e_shape = cv2.MORPH_CROSS
        # e_shape = cv2.MORPH_ELLIPSE
        e_size = 1
        e_element = cv2.getStructuringElement(e_shape, (2 * e_size + 1, 2 * e_size + 1),
                                           (e_size, e_size))

        # d_shape = cv2.MORPH_RECT
        # d_shape = cv2.MORPH_CROSS
        d_shape = cv2.MORPH_ELLIPSE
        d_size = 1
        d_element = cv2.getStructuringElement(d_shape, (2 * d_size + 1, 2 * d_size + 1),
                                           (d_size, d_size))

        img[np.abs(img) < 250] = 0
        img = cv2.erode(img, e_element, iterations=1)
        # img = cv2.dilate(img, d_element, iterations=1)

        return img

    # Functionality to find and draw contours on a frame
    def find_contours(self, image):
        ret, thresh = cv2.threshold(image, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea)

        fgmask = np.zeros_like(image)
        cv2.drawContours(fgmask, [contours[-1]], -1, (255, 255, 255), 1)

        return fgmask
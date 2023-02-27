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
            foreground = self.fgbg.apply(frame, None, 0)
            if frame is None:
                break

            # Postprocessing
            foreground[np.abs(foreground) < 254] = 0
            # foreground = cv2.erode(foreground, self.kernel, iterations=1)
            # foreground = cv2.dilate(foreground, self.kernel, iterations=1)
            fgmask = self.draw_contours(foreground)

            # Show video
            cv2.imshow('Frame', frame)
            # cv2.imshow('Unprocessed', foreground)
            cv2.imshow('Processed', fgmask)

            keyboard = cv2.waitKey(1)
            if keyboard == 'q':
                break

        video.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(0)

    # Functionality to find and draw contours on a frame
    def draw_contours(self, image):
        ret, thresh = cv2.threshold(image, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create array containing the index, area and parent of every contour and sort it in a large-small order
        contours2 = np.zeros((len(contours), 3))
        for i in range(len(contours)):
            contours2[i][0] = i
            contours2[i][1] = float(cv2.contourArea(contours[i]))
            contours2[i][2] = hierarchy[0][i][3]
        contours2 = contours2[contours2[:, 1].argsort()][::-1]

        # Sort contours in a large-small order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Make copy of the input image to draw the contours on
        image_copy = np.zeros_like(image)

        # Based on camera, set threshold for drawn contour sizes
        if self.video == 'data/cam1/video.avi':
            area = 55
        elif self.video == 'data/cam2/video.avi':
            area = 26
        elif self.video == 'data/cam3/video.avi':
            area = 137
        else:
            area = 25

        # Draw contours of the subject on the copy image
        cv2.drawContours(image_copy, [contours[0]], -1, (255, 255, 255), thickness=cv2.FILLED)
        for i in range(1, len(contours)):
            if contours2[i][1] > area:
                cv2.drawContours(image_copy, [contours[i]], -1, (0, 0, 0), thickness=cv2.FILLED)

        return image_copy
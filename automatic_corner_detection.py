"""
import cv2
import glob
import numpy as np

# Load image
img = cv2.imread("data/cam2/calibration/test_1.png")
# Apply a median blur
blur = cv2.medianBlur(img, 5)

# Find and enhance the edges
edges = cv2.Canny(blur, 200,300)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations = 1)
edges = cv2.erode(edges, kernel, iterations = 1)

# Find the contours and keep the largest (the chessboard contour)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
board = max(contours, key = cv2.contourArea)

# Find only the squares
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
stencil = np.zeros(gray.shape).astype(gray.dtype)
color = [255]
cv2.fillPoly(stencil, [board], color)
result = cv2.bitwise_and(gray, stencil)

_, thresh_inv = cv2.threshold(result, 200, 255, cv2.THRESH_BINARY_INV)

cv2.drawContours(stencil, [board], -1, 255, -1)
mask = cv2.erode(stencil, kernel, iterations = 2)

thresh_inv = cv2.bitwise_and(thresh_inv, mask)
edges = cv2.Canny(thresh_inv, 10, 50)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, 255, -1)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
import cv2
import numpy as np


img = cv2.imread("data/cam3/calibration/test_1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(img, 3)
edges = cv2.Canny(blur, 300,500)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations = 1)
edges = cv2.erode(edges, kernel, iterations = 1)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key = cv2.contourArea)

epsilon = 0.05*cv2.arcLength(c, True)
c = cv2.approxPolyDP(c,epsilon, True)

stencil = np.zeros(gray.shape).astype(gray.dtype)
color = [255]
cv2.fillPoly(stencil, [c], color)
result = cv2.bitwise_and(gray, stencil)
result = cv2.pyrUp(result)


cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
# Gaussian blur
blur = cv2.blur(img, (5, 5))

# Canny edge
edges = cv2.Canny(blur, 100, 200)

kernel = np.ones((5,5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations = 3)
edges = cv2.erode(edges, kernel, iterations = 3)
crossings = cv2.medianBlur(edges, 7)

# Corner detection
contours, hierarchy = cv2.findContours(crossings, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (0,0,255), 3)
for i in contours:
    M = cv2.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(img, (cx, cy), 7, (0, 0, 255), -1)

#result = cv2.pyrUp(result)
#result = cv2.pyrUp(result)
edges = cv2.Canny(result, 5, 50)
kernel = np.ones((5,5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations = 2)
edges = cv2.erode(edges, kernel, iterations = 1)
#edges = cv2.medianBlur(edges, 9)

edges = cv2.bitwise_not(edges)
#edges = cv2.pyrDown(edges)
#edges = cv2.pyrDown(edges)
edges = cv2.pyrDown(edges)
corners = cv2.goodFeaturesToTrack(edges, 65, 0.01, 10)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), corners, (3,3), (-1, -1), criteria)
corners = np.int0(corners)
print(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)  

    # Find the contour corners
epsilon = 0.05*cv2.arcLength(board, True)
corners = cv2.approxPolyDP(board,epsilon, True)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), corners.astype(np.float32), (5, 5), (-1, -1), criteria)
for p in corners.astype(int):
    cv2.circle(img, (p[0][0], p[0][1]), 2, 255, -1)      
        """
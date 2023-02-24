import cv2
import numpy as np


class CameraCalibrator:

    def __init__(self, n_corners, square_size, path):
        self.img = cv2.imread(path)
        self.n_corners = n_corners
        self.square_size = square_size
        self.corners = np.empty((4, 2), dtype=float)
        self.click = 0

    def get_intrinsic(self, files):
        object_points = self.prepare_object_points(self.n_corners, self.square_size)

        # Arrays to store object points and image points from all the images.
        world_points = []  # 3d point in real world space
        image_points = []  # 2d points in image plane.

        for file in files:
            # Reading the image
            img = cv2.imread(file)

            # Finding the chessboard corners
            ret, corners = cv2.findChessboardCorners(img, self.n_corners, None)

            if ret:
                world_points.append(object_points)
                image_points.append(corners)

                # cv2.drawChessboardCorners(img, (8, 6), corners, ret)
                # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                # cv2.imshow('image', img)
                # cv2.waitKey(0)

        # Performing the calibration
        ret, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(world_points, image_points,
                                                                                        img[:, :, 0].shape[::-1], None,
                                                                                        None)
        return camera_matrix, distortion_coefficients

    def get_extrinsic(self, camera_matrix, distortion_coefficients):
        self.click_outer_corner_points()
        corners = self.get_chessboard_corners((7,5), self.corners)
        object_points = self.prepare_object_points(self.n_corners, self.square_size)

        _, rvecs, tvecs = cv2.solvePnP(object_points, corners.astype(np.float32), camera_matrix,
                                       distortion_coefficients, useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE)
        return rvecs, tvecs, corners

    @staticmethod
    def prepare_object_points(n_corners, square_size):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        object_points = np.zeros((n_corners[0] * n_corners[1], 3), np.float32)
        object_points[:, :2] = np.mgrid[0:n_corners[0], 0:n_corners[1]].T.reshape(-1, 2) * square_size

        return object_points

    def click_outer_corner_points(self):
        # displaying the image
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow('image', self.img)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', self.click_event)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)

        # close the window
        cv2.destroyAllWindows()

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Saving the coordinates of the point
            corner = np.array([x, y])
            self.corners[self.click] = corner
            self.click += 1

            cv2.circle(self.img, (x, y), 1, (0, 0, 255), -1)
            cv2.imshow("image", self.img)

    def get_chessboard_corners(self, n_squares, corners):
        height = n_squares[0]*100
        width = n_squares[1]*100
        output_points = np.array([[0, 0], [0, width], [height, width], [height, 0]], dtype=np.float32)
        input_points = corners.astype(np.float32)

        M = cv2.getPerspectiveTransform(input_points, output_points)

        x = np.linspace(0, height, self.n_corners[0])
        y = np.linspace(0, width, self.n_corners[1])

        points = np.empty((self.n_corners[0]*self.n_corners[1], 2))

        n = -1
        for i in range(len(y)):
            for j in range(len(x)):
                n += 1
                points[n] = np.array([x[j], y[i]])

        image_points = cv2.perspectiveTransform(np.array([points]), np.linalg.inv(M)).astype(int)
        image_points = image_points.reshape((self.n_corners[0]*self.n_corners[1], 1, 2))
        return image_points

    @staticmethod
    def draw_chessboard_corners(img, chessboard_corners):
        for corner in chessboard_corners:
            x = corner[0][0]
            y = corner[0][1]
            cv2.circle(img, (x, y), 1, 255, -1)

        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def draw_axes(img, image_points, corner):
        img = cv2.line(img, corner, tuple(image_points[0].ravel().astype(int)), (255, 0, 0), 2)
        img = cv2.line(img, corner, tuple(image_points[1].ravel().astype(int)), (0, 255, 0), 2)
        img = cv2.line(img, corner, tuple(image_points[2].ravel().astype(int)), (0, 0, 255), 2)
        return img

    def draw_pose(self, img, r_vecs, t_vecs, camera_matrix, distortion_coef, destination):
        axes = np.float32([[4, 0, 0], [0, 4, 0], [0, 0, 4]]).reshape(-1, 3) * self.square_size

        # project 3D points to image plane
        image_points, _ = cv2.projectPoints(axes, r_vecs, t_vecs, camera_matrix, distortion_coef)

        # drawing the axes and cube
        img = self.draw_axes(img, image_points, tuple(self.corners[0].ravel().astype(int)))

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # saving the edited image
        cv2.imwrite(destination, img)



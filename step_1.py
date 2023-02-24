import cv2
from CameraCalibrator import CameraCalibrator
import glob

if __name__ == "__main__":
    cam = 1
    path = f"data/cam{cam}/calibration/test_1.png"
    calibrator = CameraCalibrator((8, 6), 115, path)

    files = glob.glob(f"data/cam{cam}/calibration/frame_*.png")
    camera_matrix, distortion_coefficients = calibrator.get_intrinsic(files)

    rotation_vectors, translation_vectors, corners = calibrator.get_extrinsic(camera_matrix,
                                                                              distortion_coefficients)

    # Visualisations
    img = cv2.imread(path)
    calibrator.draw_chessboard_corners(img, corners)

    img = cv2.imread(path)
    destination = f"data/cam{cam}/calibration/result.png"
    calibrator.draw_pose(img, rotation_vectors, translation_vectors, camera_matrix,
                         distortion_coefficients, destination)
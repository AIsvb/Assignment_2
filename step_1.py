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

    # Writing the data to XML file
    file_name = f"data/cam{cam}/config.xml"
    writer = cv2.FileStorage(file_name, cv2.FileStorage_WRITE)

    writer.write("camera_matrix", camera_matrix)
    writer.write("distortion_coefficients", distortion_coefficients)
    writer.write("rotation_vectors", rotation_vectors)
    writer.write("translation_vectors", translation_vectors)

    writer.release()

    print(rotation_vectors)
    print(translation_vectors)
    print(camera_matrix)
    print(distortion_coefficients)


    # Visualisations
    img = cv2.imread(path)
    calibrator.draw_chessboard_corners(img, corners)

    img = cv2.imread(path)
    destination = f"data/cam{cam}/calibration/result.png"
    calibrator.draw_pose(img, rotation_vectors, translation_vectors, camera_matrix,
                         distortion_coefficients, destination, cam)
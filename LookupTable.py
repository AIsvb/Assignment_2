import cv2
import numpy as np
import PIL

class LookupTable:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.voxels = []
        self.dictionary = {}
        self.cam1 = 0
        self.cam2 = 0
        self.cam3 = 0
        self.cam4 = 0
        self.configure_cameras()

        self.mask1 = cv2.imread('data/cam1/mask.png')
        self.mask2 = cv2.imread('data/cam2/mask.png')
        self.mask3 = cv2.imread('data/cam3/mask.png')
        self.mask4 = cv2.imread('data/cam4/mask.png')
        self.masks = [self.mask1, self.mask2, self.mask3, self.mask4]

    def create_voxels(self):
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    self.voxels.append(Voxel(x * 20, y * 20, z * 20, self.cam1, self.cam2, self.cam3, self.cam4))

    def create_dictionary(self):
        for v in self.voxels:
            for i in range(4):
                try:
                    self.dictionary[i + 1, (v.cam_coordinates[i][0], v.cam_coordinates[i][1])].append(v)
                except (KeyError):
                    self.dictionary[i + 1, (v.cam_coordinates[i][0], v.cam_coordinates[i][1])] = [v]

    def create_final_list(self, mask1, mask2, mask3, mask4):
        final_list = []
        for x in range(644):
            for y in range(486):
                    if mask1[y,x][0] == 255:
                        try:
                            for v in self.dictionary[1, (x,y)]:
                                v.show1 = True
                        except (KeyError):
                            0
                    if mask2[y,x][0] == 255:
                        try:
                            for v in self.dictionary[2, (x,y)]:
                                v.show2 = True
                        except (KeyError):
                            0
                    if mask3[y,x][0] == 255:
                        try:
                            for v in self.dictionary[3, (x,y)]:
                                v.show3 = True
                        except (KeyError):
                            0
                    if mask4[y,x][0] == 255:
                        try:
                            for v in self.dictionary[4, (x,y)]:
                                v.show4 = True
                        except (KeyError):
                            0

        for v in self.voxels:
            if v.show1 == True:
                if v.show2 == True:
                    if v.show3 == True:
                        if v.show4 == True:
                            final_list.append(v)

        return final_list

    # def configure_camera(self, camera):
    #     reader = cv2.FileStorage(f'data/cam{camera}/config2.xml', cv2.FileStorage_READ)
    #     r_vecs = np.array(reader.getNode("rotation_vectors"))
    #     t_vecs = np.array(reader.getNode("translation_vectors"))
    #     camera_matrix = np.array(reader.getNode("camera_matrix"))
    #     distortion_coef = np.array(reader.getNode("distortion_coefficients"))
    #
    #     return Camera(r_vecs, t_vecs, camera_matrix, distortion_coef)

    def configure_cameras(self):
        # self.r_vecs1 = np.array([[1.64067511], [0.63628855], [-0.52744192]])
        # self.t_vecs1 = np.array([[-235.8501057], [809.23960103], [4330.93372993]])
        # self.camera_matrix1 = np.array([[492.28392884, 0, 341.09828206], [0, 494.68543396, 225.08391043], [0, 0, 1]])
        # self.distortion_coef1 = np.array([[-0.39005684, 0.252636, 0.00082683, -0.00171959, -0.10728103]])
        #
        # self.r_vecs2 = np.array([[1.5342111920616499], [0.024109874315735885], [-0.026208296548183206]])
        # self.t_vecs2 = np.array([[-287.93915287816964], [1518.1110989847809], [3052.1516951094914]])
        # self.camera_matrix2 = np.array([[484.86160642438711, 0, 331.69114563515552], [0, 486.35288533453365, 221.44839834826743], [0, 0, 1]])
        # self.distortion_coef2 = np.array([[-0.32183815, 0.04770667, 0.00122654, -0.000639, 0.08593832]])
        #
        # self.r_vecs3 = np.array([[1.0954926135105636], [-1.4174370032675769], [1.3305428395560470]])
        # self.t_vecs3 = np.array([[-873.58324378684506], [1249.0493039984979], [3197.8182916152950]])
        # self.camera_matrix3 = np.array([[484.52081847, 0, 315.39239273], [0, 482.61964346, 228.89477142], [0, 0, 1]])
        # self.distortion_coef3 = np.array([[-0.36574808, 0.18783658, 0.00264871, 0.00152216 - 0.06609458]])
        #
        # self.r_vecs4 = np.array([[1.5682182743826338], [-0.73149678961419984], [-0.65660363186292048]])
        # self.t_vecs4 = np.array([[-516.46375670837551], [999.03474066008425], [375.25520114715955]])
        # self.camera_matrix4 = np.array([[486.79378735109771, 0, 341.57350589262217], [0, 489.86418274652539, 239.93430544330724], [0, 0, 1]])
        # self.distortion_coef4 = np.array([[-0.37637834, 0.21504516, -0.00046899, -0.00127208, -0.07991701]])

        self.r_vecs1 = np.array([[1.64067511], [0.63628855], [-0.52744192]])
        self.t_vecs1 = np.array([[-235.8501057], [809.23960103], [4330.93372993]])
        self.camera_matrix1 = np.array([[492.28392884, 0, 341.09828206], [0, 494.68543396, 225.08391043], [0, 0, 1]])
        self.distortion_coef1 = np.array([[-0.39005684, 0.252636, 0.00082683, -0.00171959, -0.10728103]])

        self.r_vecs2 = np.array([[1.53077262], [0.01779587], [-0.02396146]])
        self.t_vecs2 = np.array([[-292.9776613], [1504.31527588], [3035.83510593]])
        self.camera_matrix2 = np.array([[484.86160645, 0, 331.69114576], [0, 486.35288535, 221.44839849], [0, 0, 1]])
        self.distortion_coef2 = np.array([[-0.32183815, 0.04770667, 0.00122654, -0.000639, 0.08593832]])

        self.r_vecs3 = np.array([[1.10116539], [-1.40502394], [1.33760412]])
        self.t_vecs3 = np.array([[-125.01790695], [1242.21133861], [2516.27419716]])
        self.camera_matrix3 = np.array([[484.52081847, 0, 315.39239273], [0, 482.61964346, 228.89477142], [0, 0, 1]])
        self.distortion_coef3 = np.array([[-0.36574808, 0.18783658, 0.00264871, 0.00152216 - 0.06609458]])

        self.r_vecs4 = np.array([[1.57141214], [-0.73216362], [0.6475565]])
        self.t_vecs4 = np.array([[-519.29162894], [1017.69518996], [3830.09293964]])
        self.camera_matrix4 = np.array([[486.79378731, 0, 341.57350559], [0, 489.86418276, 239.93430545], [0, 0, 1]])
        self.distortion_coef4 = np.array([[-0.37637834, 0.21504516, -0.00046899, -0.00127208, -0.07991701]])

        self.cam1 = Camera(self.r_vecs1, self.t_vecs1, self.camera_matrix1, self.distortion_coef1)
        self.cam2 = Camera(self.r_vecs2, self.t_vecs2, self.camera_matrix2, self.distortion_coef2)
        self.cam3 = Camera(self.r_vecs3, self.t_vecs3, self.camera_matrix3, self.distortion_coef3)
        self.cam4 = Camera(self.r_vecs4, self.t_vecs4, self.camera_matrix4, self.distortion_coef4)


class Voxel:
    def __init__(self, width, height, depth, cam1, cam2, cam3, cam4):
        self.voxel_coordinates = (width, height, depth)
        self.cam1_coordinates, _ = cv2.projectPoints(self.voxel_coordinates, cam1.r_vecs, cam1.t_vecs, cam1.camera_matrix, cam1.distortion_coef)
        self.cam2_coordinates, _ = cv2.projectPoints(self.voxel_coordinates, cam2.r_vecs, cam2.t_vecs, cam2.camera_matrix, cam2.distortion_coef)
        self.cam3_coordinates, _ = cv2.projectPoints(self.voxel_coordinates, cam3.r_vecs, cam3.t_vecs, cam3.camera_matrix, cam3.distortion_coef)
        self.cam4_coordinates, _ = cv2.projectPoints(self.voxel_coordinates, cam4.r_vecs, cam4.t_vecs, cam4.camera_matrix, cam4.distortion_coef)

        self.cam_coordinates = [(int(self.cam1_coordinates[0][0][0]),int(self.cam1_coordinates[0][0][1])), (int(self.cam2_coordinates[0][0][0]),int(self.cam2_coordinates[0][0][1])),
                                (int(self.cam3_coordinates[0][0][0]), int(self.cam3_coordinates[0][0][1])), (int(self.cam4_coordinates[0][0][0]),int(self.cam4_coordinates[0][0][1]))]
        self.show1 = False
        self.show2 = False
        self.show3 = False
        self.show4 = False
class Camera:
    def __init__(self, r_vecs, t_vecs, camera_matrix, distortion_coef):
        self.r_vecs = r_vecs
        self.t_vecs = t_vecs
        self.camera_matrix = camera_matrix
        self.distortion_coef = distortion_coef


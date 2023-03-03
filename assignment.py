import glm
import random
import numpy as np
import cv2
from LookupTable import LookupTable as LT


# Masks to start off with
mask1 = cv2.imread('data/cam1/mask.png')
mask2 = cv2.imread('data/cam2/mask2_dilate3.png')
mask3 = cv2.imread('data/cam3/mask.png')
mask4 = cv2.imread('data/cam4/mask.png')

# Create a look-up table
LT = LT(50, 75, 100, mask1, mask2, mask3, mask4)
LT.create_voxels()
LT.create_lookup_table()

# Function to get all voxels that need to be shown
list = LT.get_voxels(mask1, mask2, mask3, mask4)

block_size = 1.0

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1, 1, 1])
    return data, colors

def set_voxel_positions():
    data, colors = [], []

    # mask1 = cv2.imread(m1)
    # mask2 = cv2.imread(m2)
    # mask3 = cv2.imread(m3)
    # mask4 = cv2.imread(m4)

    # for v in LT.get_voxels(mask1, mask2, mask3, mask4):
    for v in list:
        data.append([v.voxel_coordinates[0] * 0.05 - 8, v.voxel_coordinates[2] * 0.05, v.voxel_coordinates[1] * 0.05 - 10])
        colors.append([v.color[0], v.color[1], v.color[2]])
    return data, colors

def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    v = []
    for i in range(4):
        cam = i + 1
        file_name = f"data/cam{cam}/config.xml"

        # Reading from XML
        reader = cv2.FileStorage(file_name, cv2.FileStorage_READ)

        t_vecs = reader.getNode("translation_vectors").mat()
        r_vecs = reader.getNode("rotation_vectors").mat()
        r_vecs, _ = cv2.Rodrigues(r_vecs)

        t_vecs = -np.matrix(r_vecs).T * np.matrix(t_vecs)

        t_vecs = t_vecs.astype(int)
        translation= [-(t_vecs[0,0]-500)/20, t_vecs[2,0]/20, t_vecs[1,0]/20]

        v.append(translation)
    return v, \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]

def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = get_rotation_matrices()
    return cam_rotations

def get_rotation_matrices():
    M = []
    for i in range(4):
        cam = i + 1
        file_name = f"data/cam{cam}/config.xml"

        # Reading from XML
        reader = cv2.FileStorage(file_name, cv2.FileStorage_READ)

        r_vecs = reader.getNode("rotation_vectors").mat()
        r_vecs = cv2.Rodrigues(r_vecs)

        I = np.identity(4, dtype=np.float64)
        for i in range(3):
            for j in range(3):
                I[i, j] = r_vecs[0][i, j]

        mat = glm.mat4(I)
        mat = glm.rotate(mat, glm.radians(90), (0, 0, 1))
        M.append(mat)
    return M
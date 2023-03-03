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
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]

def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
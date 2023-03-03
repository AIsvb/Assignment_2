import glm
import random
import numpy as np
import cv2
from LookupTable import LookupTable as LT
import PIL

mask1 = cv2.imread('data/cam1/mask.png')
mask2 = cv2.imread('data/cam2/mask.png')
mask3 = cv2.imread('data/cam3/mask.png')
mask4 = cv2.imread('data/cam4/mask.png')


LT = LT(50, 50, 100)
LT.create_voxels()
LT.create_dictionary()
list = LT.create_final_list(mask1, mask2, mask3, mask4)

block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0,1.0,1.0])
    return data, colors


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    data, colors = [], []
    for v in list:
        data.append([v.voxel_coordinates[0] * 0.02 - 8, v.voxel_coordinates[2] * 0.02, v.voxel_coordinates[1] * 0.02 - 10])
        colors.append([0.5, 0.5, 0.5])
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
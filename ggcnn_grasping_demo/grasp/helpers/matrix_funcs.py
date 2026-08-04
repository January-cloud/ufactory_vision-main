import numpy as np
import math

def get_sign(x):
    if x>=0:
        return 1
    return -1
    
# Transfer from given rpy_angles:[roll, pitch, yaw] to rotation matrix Rot
def rpy_to_rot(rpy_angles):
    # Rx = np.mat([[1, 0, 0],[ 0, math.cos(rpy_angles[0]), -math.sin(rpy_angles[0])],[ 0, math.sin(rpy_angles[0]), math.cos(rpy_angles[0])]])
    # Ry = np.mat([[math.cos(rpy_angles[1]),0, math.sin(rpy_angles[1])],[ 0, 1, 0],[ -math.sin(rpy_angles[1]), 0, math.cos(rpy_angles[1])]])
    # Rz = np.mat([[math.cos(rpy_angles[2]), -math.sin(rpy_angles[2]), 0],[ math.sin(rpy_angles[2]), math.cos(rpy_angles[2]), 0],[ 0, 0, 1]])
    
    Rx = np.asmatrix([[1, 0, 0],[ 0, math.cos(rpy_angles[0]), -math.sin(rpy_angles[0])],[ 0, math.sin(rpy_angles[0]), math.cos(rpy_angles[0])]])
    Ry = np.asmatrix([[math.cos(rpy_angles[1]),0, math.sin(rpy_angles[1])],[ 0, 1, 0],[ -math.sin(rpy_angles[1]), 0, math.cos(rpy_angles[1])]])
    Rz = np.asmatrix([[math.cos(rpy_angles[2]), -math.sin(rpy_angles[2]), 0],[ math.sin(rpy_angles[2]), math.cos(rpy_angles[2]), 0],[ 0, 0, 1]])

    Rot = Rz*Ry*Rx
    return Rot

# Transfer from given rotation matrix RotM to [roll, pitch, yaw] angles
def rot_to_rpy(RotM):
    pitch = math.atan2(-RotM[2,0], math.sqrt(RotM[2,1]**2 + RotM[2,2]**2))
    sign = get_sign(math.cos(pitch))
    roll = math.atan2(RotM[2,1]*sign, RotM[2,2]*sign)
    yaw = math.atan2(RotM[1,0]*sign, RotM[0,0]*sign)
    return [roll, pitch, yaw]

def euler2mat(euler_vect6d):
    rpy = euler_vect6d[3:]
    rot_3x3 = rpy_to_rot(rpy)
    xyz = np.array([[euler_vect6d[0]],[euler_vect6d[1]],[euler_vect6d[2]]])
    padding_row = np.array([[0,0,0,1]])
    tmp_mat = np.concatenate((rot_3x3,xyz), axis=1); # shape is now (3,4)
    mat4x4 = np.concatenate((tmp_mat, padding_row), axis=0) # shape is now (4,4)
    return mat4x4 # still matrix

def mat2euler(mat4x4):
    rot = mat4x4[0:3, 0:3]
    xyz = mat4x4[:3, 3]
    rpy = rot_to_rpy(rot) #list
    return [xyz[0].item(), xyz[1].item(),xyz[2].item(),rpy[0], rpy[1], rpy[2]]

# mat_of_source_in_tar_frame: source frame expressed in target frame
def convert_pose(pose6d, mat_of_source_in_tar_frame):
    mat_in_source = euler2mat(pose6d)
    mat_in_tar = mat_of_source_in_tar_frame * mat_in_source
    return mat2euler(mat_in_tar)

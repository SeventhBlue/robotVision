#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/8/17 14:34
# @Author      :weiz
# @ProjectName :robotVision2
# @File        :common.py
# @Description :Universal function
import numpy as np
import cv2


def transfer_pixel2camera_3d_point(u, v, depth, params):
    """
    transfer pixel coordinates to camera coordinates.
    :param u: 2d point of object x pixel
    :param v: 2d point of object y pixel
    :param depth: point depth(m)
    :param params: camera intrinsic parameters
    :return:
    """
    camera_3D = []

    camera_3D.append(depth * (u - params[0][2]) / params[0][0])
    camera_3D.append(depth * (v - params[1][2]) / params[1][1])
    camera_3D.append(depth)

    return camera_3D


def transfer_camera2gripper_3d_point(camera_3D, R_camera2gripper, t_camera2gripper):
    """
    transfer camera coordinates to gripper coordinates.
    :param camera_3D:
    :param R_camera2gripper:
    :param t_camera2gripper:
    :return:
    """
    camera_3D = np.array(camera_3D)
    r = np.array(R_camera2gripper)
    t = np.array(t_camera2gripper)
    gripper_3D = np.matmul(camera_3D, r.T)
    gripper_3D = gripper_3D + t

    return gripper_3D


def transfer_gripper2base_arm_3d_point(gripper_3D, gripper_posture):
    """
    transfer gripper coordinates to the base of robot arm coordinates.
    :param gripper_3D:
    :param gripper_posture: [x, y, z, rx, ry, rz]
    :return:
    """
    R_gripper2base = np.zeros((3, 3))
    cv2.Rodrigues(np.array([gripper_posture[3], gripper_posture[4], gripper_posture[5]]), R_gripper2base)
    base_arm_position = np.matmul(gripper_3D, R_gripper2base.T) + \
                        np.array([gripper_posture[0], gripper_posture[1], gripper_posture[2]])

    return base_arm_position


def get_target2base_posture(R_target2camera_vector, R_camera2gripper, R_gripper2base_vector):
    """
    get the posture of the target in the base of robot arm coordinates
    :param R_target2camera_vector:
    :param R_camera2gripper:
    :param R_gripper2base_vector:
    :return:
    """
    R_target2camera = np.zeros((3, 3), dtype=np.float64)
    R_gripper2base = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(R_target2camera_vector, R_target2camera)
    cv2.Rodrigues(R_gripper2base_vector, R_gripper2base)

    R = np.matmul(R_target2camera, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))  # aruco marker
    R = np.matmul(R_camera2gripper, R)
    R_target2base = np.matmul(R_gripper2base, R)

    R_target2base_vector = np.zeros((3, 1), dtype=np.float64)
    cv2.Rodrigues(R_target2base, R_target2base_vector)
    return R_target2base_vector


def read_line(file_name):
    """
    read file by line
    :param file_name:
    :return:
    """
    lines = []
    file = open(file_name)
    for line in file:
        line = line.strip('\n')
        lines.append(line)
    file.close()
    return lines
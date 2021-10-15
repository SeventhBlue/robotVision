#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/10/12 16:49
# @Author      :weiz
# @ProjectName :robotVision3
# @File        :put_position.py
# @Description :
# Copyright (C) 2021-2025 Jiangxi Institute Of Intelligent Industry Technology Innovation
from agile_robot import AgileRobot
from zed2_camera import Zed2Camera

import copy
from common import *


def get_aruco_marker_center(img):
    """
    get center point of aruco marker
    :param img:
    :return:
    """
    img_copy = copy.deepcopy(img)

    img_copy = 255 - img_copy
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    # Specify the specification of the aruco marker to be identified
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    # lists of ids and the corners beloning to each id
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict,
                                              parameters=cv2.aruco.DetectorParameters_create())

    if ids is not None:
        center_list = []
        for ind, val in enumerate(ids):
            corners_id = np.squeeze(corners[ind])
            x_min = min(corners_id[0][0], corners_id[1][0], corners_id[2][0], corners_id[3][0])
            x_max = max(corners_id[0][0], corners_id[1][0], corners_id[2][0], corners_id[3][0])
            y_min = min(corners_id[0][1], corners_id[1][1], corners_id[2][1], corners_id[3][1])
            y_max = max(corners_id[0][1], corners_id[1][1], corners_id[2][1], corners_id[3][1])

            x_center = int((x_min + x_max) / 2)
            y_center = int((y_min + y_max) / 2)
            # cv2.circle(img, (x_center, y_center), 2, (0, 255, 0), 2)
            # cv2.imshow("show center points", img)
            center_list.append([x_center, y_center])

        return True, center_list
    else:
        return False, []


def get_put_position(img, cam, arm_robot, R_camera2gripper, t_camera2gripper):
    """

    :param img:
    :param cam:
    :param arm_robot:
    :param R_camera2gripper:
    :param t_camera2gripper:
    :return:
    """
    is_ok, center_list = get_aruco_marker_center(img)
    if is_ok:
        points_3d = []
        for [x, y] in center_list:
            # get 3D point of camera
            ipm, dpv = cam.get_ipm_dpv()
            depth_object = cam.get_depth_value(x, y)
            camera_3D_object = transfer_pixel2camera_3d_point(x, y, depth_object, ipm)

            # get 3D point of gripper
            griper_position_object = transfer_camera2gripper_3d_point(camera_3D_object, R_camera2gripper,
                                                                      t_camera2gripper)

            # Get the posture of the end of the robotic arm
            gripper_posture = arm_robot.get_robotic_arm_posture()

            # get 3D point of the base of robotic arm
            base_arm_3D_object = transfer_gripper2base_arm_3d_point(griper_position_object, gripper_posture)
            points_3d.append(base_arm_3D_object)

        return True, points_3d
    else:
        return False, []


def main():
    cap = cv2.VideoCapture(0)
    cam = Zed2Camera(depth_mode="PERFORMANCE")
    arm_robot = AgileRobot("192.168.10.167")

    R_camera2gripper = [[0.00719635, 0.99956883, 0.02846701],
         [-0.99982521, 0.00670111, 0.01745426],
         [0.01725598, -0.02858765, 0.99944233]]

    t_camera2gripper = [-0.07718228,  0.07114343, 0.00472535]

    while True:
        # ret, frame = cap.read()
        frame = cam.get_opencv_format()

        is_ok, point_3d = get_put_position(frame, cam, arm_robot, R_camera2gripper, t_camera2gripper)
        if is_ok:
            for val in point_3d:
                print(val)
        else:
            print("Flase")

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
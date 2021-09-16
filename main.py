#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/7/12 17:08
# @Author      :weiz
# @ProjectName :robotVision2
# @File        :main.py
# @Description :
from calibrate import Calibrate
from zed2_camera import Zed2Camera
from common import *
from universal_robot import UR
from agile_robot import AgileRobot
from pose_detection import PoseDetection
import time
import os

def save_data(file_name, data):
    """
    Format and save data
    :param file_name:
    :param data:
    :return:
    """
    info_file = open(file_name, mode='a')
    info_file.write('[')
    for ind, line in enumerate(data):
        info_file.write('[')
        print("line:", line)
        for i, value in enumerate(line):
            if i > (len(line) - 2):
                if ind < len(data) - 1:
                    info_file.write(str(value) + ']' + ',' + '\n')
                else:
                    info_file.write(str(value) + ']')

            else:
                info_file.write(str(value) + ", ")
    info_file.write(']' + '\n' + '\n')

    info_file.close()


def calibrate(cam, arm_robot, cb_size, side_length, hand_eye_img_path):
    """
    calibrate
    :param cam:
    :param arm_robot:
    :param cb_size:
    :param side_length:
    :param hand_eye_img_path:
    :return:
    """
    # Collect hand-eye calibration images and the pose of the robotic arm
    # cam.show_img()
    robotic_arm_posture = cam.save_img_key(arm_robot, hand_eye_img_path)
    for pose_gripper in robotic_arm_posture:
        print(pose_gripper)

    # get R t
    calibrate = Calibrate(cb_size, side_length)
    ipm, dpv = cam.get_ipm_dpv()
    R_camera2gripper, t_camera2gripper = calibrate.calibrate_hand_in_eye(robotic_arm_posture, ipm, dpv, hand_eye_img_path)
    info_file_name = os.path.join(hand_eye_img_path, time.strftime("%Y_%m_%d_%H_%M_%S.txt", time.localtime()))
    save_data(info_file_name, robotic_arm_posture)
    save_data(info_file_name, R_camera2gripper)
    save_data(info_file_name, t_camera2gripper)

    return R_camera2gripper, t_camera2gripper


def guide_robotic_arm(object_pixel_coord, arm_robot, cam, pose_detection, R_camera2gripper, t_camera2gripper):
    """
    Guide the robotic arm to grab
    :param object_pixel_coord:
    :param arm_robot:
    :param cam:
    :param pose_detection:
    :param R_camera2gripper:
    :param t_camera2gripper:
    :return:
    """
    cam.show_img()

    # get 3D point of camera
    ipm, dpv = cam.get_ipm_dpv()
    depth_object = cam.get_depth_value(object_pixel_coord[0], object_pixel_coord[1])
    camera_3D_object = transfer_pixel2camera_3d_point(object_pixel_coord[0], object_pixel_coord[1], depth_object, ipm)
    print("get depth is:", depth_object)
    print("camera 3d point is :", camera_3D_object)

    # get 3D point of gripper
    griper_position_object = transfer_camera2gripper_3d_point(camera_3D_object, R_camera2gripper, t_camera2gripper)
    print("gripper position 3d point is :", griper_position_object)

    # Get the posture of the end of the robotic arm
    gripper_posture = arm_robot.get_robotic_arm_posture()

    # get 3D point of the base of robotic arm
    base_arm_3D_object = transfer_gripper2base_arm_3d_point(griper_position_object, gripper_posture)
    print("base arm 3d point is :", base_arm_3D_object)
    print(base_arm_3D_object)

    # Get the posture of the detected target
    img = cam.get_opencv_format()
    R_vector_posture, t_vector, img, euler_angles = pose_detection.get_id_marker_rotation_vector(img, ipm, dpv, 0.04, 6)
    R_vector_posture_base = get_target2base_posture(R_vector_posture, R_camera2gripper,
                                                    [gripper_posture[3], gripper_posture[4], gripper_posture[5]])

    # guide the robotic arm to grab
    # arm_robot.send_order_gripper([base_arm_3D_object[0], base_arm_3D_object[1], base_arm_3D_object[2],
    #                       R_vector_posture_base[0], R_vector_posture_base[1], R_vector_posture_base[2]])


cb_size = (6, 8)     # (6, 8)
side_length = 0.035    # (0.035)
hand_eye_img_path = "./test8"
object_u_v = [777, 450]
port = 30003
host = "192.168.10.75"  # 192.168.10.167   192.168.207.128
buf = 1140
def main():
    cam = Zed2Camera(depth_mode="PERFORMANCE")
    # arm_robot = UR(host, port, buf)
    arm_robot = AgileRobot(host)
    R_camera2gripper, t_camera2gripper = calibrate(cam, arm_robot, cb_size, side_length, hand_eye_img_path)
    print(R_camera2gripper)
    print(t_camera2gripper)
    
    # R_camera2gripper = [[0.00719635, 0.99956883, 0.02846701],
    #      [-0.99982521, 0.00670111, 0.01745426],
    #      [0.01725598, -0.02858765, 0.99944233]]
    #
    # t_camera2gripper = [-0.07718228,  0.07114343, 0.00472535]

    # pose_detection = PoseDetection(cb_size)
    # guide_robotic_arm(object_u_v, arm_robot, cam, pose_detection, R_camera2gripper, t_camera2gripper)

    # calibrate = Calibrate(cb_size, side_length)
    # ipm, dpv = cam.get_ipm_dpv()
    # R, t = calibrate.calibrate_offline("./test3", "./2021_08_26_16_05_31.txt", ipm, dpv)
    # print(R)
    # print(t)

if __name__ == "__main__":
    main()
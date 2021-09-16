#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/7/20 14:08
# @Author      :chentao
# @ProjectName :robotvision3
# @File        :main_realtime.py
# @Description :
# Copyright?(C) 2021-2025?Jiangxi Institute Of Intelligent Industry Technology Innovation
import sys
sys.path.append('/home/sky/Project/yolov5-jetson/build')
import cv2
import math
import _thread
import numpy as np
import yolov5_module
from universal_robot import UR
from agile_robot import AgileRobot
from   calibrate import Calibrate
from pose_detection import PoseDetection
from   zed2_camera import Zed2Camera
from   common import *


def get_origin_target_rect(img_h, img_w, dect_h, dect_w, bbox):
    r_w = dect_w / (img_w * 1.0)
    r_h = dect_h / (img_h * 1.0)
    if (r_h > r_w) :
        l = bbox[0] - bbox[2]/2.0
        r = bbox[0] + bbox[2]/2.0
        t = bbox[1] - bbox[3]/2.0 - (dect_h - r_w * img_h) / 2
        b = bbox[1] + bbox[3]/2.0 - (dect_h - r_w * img_h) / 2
        l = l / r_w
        r = r / r_w
        t = t / r_w
        b = b / r_w
    else :
        l = bbox[0] - bbox[2]/2.0 - (dect_w - r_h * img_w) / 2
        r = bbox[0] + bbox[2]/2.0 - (dect_w - r_h * img_w) / 2
        t = bbox[1] - bbox[3]/2.0
        b = bbox[1] + bbox[3]/2.0
        l = l / r_h
        r = r / r_h
        t = t / r_h
        b = b / r_h
    
    return np.array([l, t, r-l, b-t])


def get_detect_object_uv(img, model, labels, label, model_h, model_w):
    inference_ret = model.image_inference(img)

    u = 0
    v = 0
    conf = 0

    for ret in inference_ret:
        rect = get_origin_target_rect(img.shape[0], img.shape[1], model_w, model_h, ret.bbox)

        p1 = (int(rect[0]), int(rect[1]))
        p2 = (int(rect[0] + rect[2]), int(rect[1] + rect[3]))
        cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
        cv2.putText(img, labels[int(ret.classid)]+ ' ' + str(int(ret.conf * 100)), p1,
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
        
        if (labels[int(ret.classid)] == label) and (ret.conf > conf):
            u = int(round(rect[0] + rect[2] / 2.0))
            v = int(round(rect[1] + rect[3] / 2.0))
            conf = ret.conf

    
    return (u, v), img


def get_b_point(arm_robot, cam, R_camera2gripper, t_camera2gripper, object_uv, depth, pose_detect=None):
    # get 3D point of camera
    ipm, dpv = cam.get_ipm_dpv()
    camera_3D_object = transfer_pixel2camera_3d_point(object_uv[0], object_uv[1], depth, ipm)

    # get 3D point of gripper
    griper_position_object = transfer_camera2gripper_3d_point(camera_3D_object, R_camera2gripper, t_camera2gripper)

    # Get the posture of the end of the robotic arm
    gripper_posture = arm_robot.get_robotic_arm_posture()

    # get 3D point of the base of robotic arm
    base_arm_3D_object = transfer_gripper2base_arm_3d_point(griper_position_object, gripper_posture)

    if pose_detect:
        # Get the posture of the detected target
        img = cam.get_opencv_format()
        R_vector_posture, t_vector, img, euler_angles = pose_detection.get_id_marker_rotation_vector(img, ipm, dpv, 0.04, 6)
        R_vector_posture_base = get_target2base_posture(R_vector_posture, R_camera2gripper,
                                                        [gripper_posture[3], gripper_posture[4], gripper_posture[5]])

        return [base_arm_3D_object[0], base_arm_3D_object[1], base_arm_3D_object[2],
                R_vector_posture_base[0], R_vector_posture_base[1], R_vector_posture_base[2]]
    else:
        return [base_arm_3D_object[0], base_arm_3D_object[1], base_arm_3D_object[2],
                gripper_posture[3], gripper_posture[4], gripper_posture[5]] 



cb_size = (6, 8)
side_length = 0.035
A_point = [0.52636, 0.45840, 0.49114, 0.25227, -0.10777, 0.35207]
C_point = [0.39975, 0.78972, 0.05668, 0.02058, -0.02605, 0.54221]
ip = "192.168.10.75"
port = 30003
buf = 1140
engine_path = "/home/sky/Project/yolov5-jetson/build/yolov5s.engine"
def main():
    cam = Zed2Camera()
    arm_robot = AgileRobot(ip)
    pose_detect = PoseDetection(cb_size)
    yolov5_module.init_inference(engine_path)
    labels = read_line("test.names")

    R_camera2gripper = [[0.016698394071080716, -0.9969513464305503, -0.07621795382821285],
                        [-0.9942400506134641, -0.024627453851650206, 0.1043082464281741],
                        [-0.10586730086094912, 0.0740371620680962, -0.9916201960636535]]
    t_camera2gripper = [0.11636480567993772, -0.02366999993215884, 0.07551712402327147]

    # A point start
    arm_robot.send_order_gripper(A_point)
    depth_error = 0
    time_span = 0
    while True:
        img = cam.get_opencv_format()

        # get u,v, depth
        object_uv, detect_ret_img = get_detect_object_uv(img, yolov5_module, labels, "steel box", 608, 608)
        depth = cam.get_depth_value(object_uv[1], object_uv[0])  #compensate for error
        if (math.isnan(depth) or math.isinf(depth) or depth < 0.3 or depth > 2.0):
            depth_error = depth_error + 1
            if depth_error % 50 == 0:
                arm_robot.send_order_gripper(A_point)
            print("depth:", depth)
            continue
        
        # guide arm robot to B point
        B_point = get_b_point(arm_robot, cam, R_camera2gripper, t_camera2gripper, object_uv, depth)
        B_point[2] = 0.030
        arm_robot.send_order_gripper(B_point)
        arm_robot.send_order_digital(0, 8)
        cv2.waitKey(1000)

        # back to A point
        arm_robot.send_order_gripper(A_point)
        
        # C point
        arm_robot.send_order_gripper(C_point)
        arm_robot.send_order_digital(0, 0)
        cv2.waitKey(1000)

        depth_error = 0

        cv2.imshow("detect_ret_img", detect_ret_img)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break



if __name__ == "__main__":
    main()
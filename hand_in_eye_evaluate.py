#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/9/15 9:42
# @Author      :weiz
# @ProjectName :coco_evaluate
# @File        :hand_in_eye_evaluate.py
# @Description :
import sys
sys.path.append('/home/sky/Project/yolov5-jetson/build')
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yolov5_module
from universal_robot import UR
from agile_robot import AgileRobot
from calibrate import Calibrate
from pose_detection import PoseDetection
from zed2_camera import Zed2Camera
from common import *


def draw_circle():
    """
    draw circle
    :return:
    """
    img = np.ones((1000, 1000, 3), np.uint8) * 255

    point_size = 1
    red_color = (0, 0, 255)
    black_color = (0, 0, 0)

    thickness = 4

    cv2.circle(img, (500, 500), point_size, red_color, thickness)
    cv2.circle(img, (500, 500), 450, red_color, 30)

    cv2.circle(img, (500, 500), 50, black_color, 1)
    cv2.circle(img, (500, 500), 100, black_color, 1)
    cv2.circle(img, (500, 500), 150, black_color, 1)
    cv2.circle(img, (500, 500), 200, black_color, 1)
    cv2.circle(img, (500, 500), 250, black_color, 1)
    cv2.circle(img, (500, 500), 300, black_color, 1)
    cv2.circle(img, (500, 500), 350, black_color, 1)
    cv2.circle(img, (500, 500), 400, black_color, 1)

    cv2.imshow('image', img)
    cv2.imwrite("circle.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def circle_detect(img):
    """
    Detect circle and return center coordinate
    :param img:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh1', thresh1)

    canny = cv2.Canny(thresh1, 40, 80)
    # cv2.imshow('Canny', canny)

    canny = cv2.blur(canny, (3, 3))
    # cv2.imshow('blur', canny)

    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=60, minRadius=30, maxRadius=200)

    if circles is not None and len(circles) == 1:
        for circle in circles[0]:
            x = int(circle[0])
            y = int(circle[1])
            r = int(circle[2])

            img = cv2.circle(img, (x, y), r, (0, 0, 255), 2)
        cv2.imshow('ret', img)
        return (x, y), img
    else:
        return (0, 0), img


def red_extraction(img):
    """
    Detect and return circle center coordinates through color extraction
    :param img:
    :return:
    """
    hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_low_range_1 = np.array([156, 43, 46])
    green_high_range_1 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hue_image, green_low_range_1, green_high_range_1)

    green_low_range_2 = np.array([0, 43, 46])
    green_high_range_2 = np.array([10, 255, 255])
    mask2 = cv2.inRange(hue_image, green_low_range_2, green_high_range_2)

    mask = mask1 + mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    cx = 0
    cy = 0
    if len(contours) > 0:
        for contour in contours:
            if cv2.contourArea(contour) > max_area:
                # M = cv2.moments(contour)
                # cx = int(M['m10'] / (M['m00'] * 1.0))
                # cy = int(M['m01'] / (M['m00'] * 1.0))

                [x, y, w, h] = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2

                max_area = cv2.contourArea(contour)
                cv2.circle(mask, (cx, cy), 1, (255, 255, 255), 4)

    return (cx, cy), mask


def get_object_coord(arm_robot, cam, R_camera2gripper, t_camera2gripper, object_uv):
    """
    Pixel coordinates to robot base coordinates
    :param arm_robot:
    :param cam:
    :param R_camera2gripper:
    :param t_camera2gripper:
    :param object_uv:
    :return:
    """
    # get 3D point of camera
    ipm, dpv = cam.get_ipm_dpv()
    depth = cam.get_depth_value(object_uv[1], object_uv[0])  # compensate for error
    camera_3D_object = transfer_pixel2camera_3d_point(object_uv[0], object_uv[1], depth, ipm)

    # get 3D point of gripper
    griper_position_object = transfer_camera2gripper_3d_point(camera_3D_object, R_camera2gripper, t_camera2gripper)

    # Get the posture of the end of the robotic arm
    gripper_posture = arm_robot.get_robotic_arm_posture()

    # get 3D point of the base of robotic arm
    base_arm_3D_object = transfer_gripper2base_arm_3d_point(griper_position_object, gripper_posture)

    return [base_arm_3D_object[0], base_arm_3D_object[1], base_arm_3D_object[2]]


def save_points(data, magnification=1):
    """
    Save error data and show analysis
    :param data:
    :param magnification: [[x, y, z],...]
    :return:
    """
    data_array = np.array(data)
    mean = np.mean(data_array, axis=0)
    std = data_array.std(axis=0)
    file_name = "mean_{:.2f}-{:.2f}-{:.2f}".format((mean[0] - data_array[0][0]) * 1000,
                                                   (mean[1] - data_array[0][1]) * 1000,
                                                   (mean[2] - data_array[0][2]) * 1000)
    print("mean:", mean - data_array[0])
    print("std:", std)

    f = open(file_name + ".txt", mode='a')
    for line in data:
        f.write(str(line) + '\n')
    f.write('mean_error:' + str(mean - data_array[0]) + '\n')
    f.write('std_error:' + str(std) + '\n')
    f.close()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    
    X = (data_array[:, 0] - data_array[0][0]) * magnification
    Y = (data_array[:, 1] - data_array[0][1]) * magnification
    Z = (data_array[:, 2] - data_array[0][2]) * magnification

    ax.scatter(X, Y, Z, alpha=0.3, c=np.random.random(len(X)), s=np.random.randint(10, 20, size=(20, 40)), marker="^", label="Error data")
    ax.scatter([0], [0], [0], alpha=0.3, c="#FF0000", s=30, label="Origin")

    ax.legend()
    plt.title("Hand-eye calibration error distribution magnified {} Times".format(magnification), fontsize=16)
    ax.set_xlabel("X label")
    ax.set_ylabel("Y label")
    ax.set_zlabel("Z label")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())

    plt.savefig(file_name + ".png", dpi=600)

    plt.show()


cb_size = (6, 8)
side_length = 0.035
ip = "192.168.10.75"
port = 30003
buf = 1140
engine_path = "/home/sky/Project/yolov5-jetson/build/yolov5s.engine"
def main():
    cam = Zed2Camera()
    arm_robot = AgileRobot(ip)
    yolov5_module.init_inference(engine_path)

    R_camera2gripper = [[0.16827259605935702, 0.8697706156883513, 0.46387865816455864],
                        [0.912724292859175, -0.31521980331685084, 0.25994391860862054],
                        [0.37231532152217917, 0.3796518822315952, -0.8469036247872859]]
    t_camera2gripper = [-0.1091670905222476,-0.01973218527775718,0.012206317139845681]
    # arm_robot.connect_robotic_arm()
    # [x_real, y_real, z_real, _, _, _] = arm_robot.get_robotic_arm_posture()
    # arm_robot.connect_robotic_arm()

    #recorded real point
    [x_real, y_real, z_real, _, _, _] = [0.5574254116635314, 0.5254123046717327, 0.0025159556206567485, 0.3474143306151003, 0.6995536077763843, -2.1571154792347795]

    detect_points = []
    detect_points.append([x_real, y_real, z_real])
    while True:
        img = cam.get_opencv_format()

        # get u, v
        # object_uv, detect_ret_img = circle_detect(img)
        object_uv, detect_ret_img = red_extraction(img)

        # get the world coordinates of the target
        x, y, z = get_object_coord(arm_robot, cam, R_camera2gripper, t_camera2gripper, object_uv)
        if (math.isnan(x) or math.isinf(x) or math.isnan(y) or math.isinf(y) or math.isnan(z) or math.isinf(z)):
            continue
        detect_points.append([x, y, z])

        cv2.putText(img, "Number of error data: " + str(len(detect_points)), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, "Black bullseye: real point", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, "gree point: detect point", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.circle(img, object_uv, 1, (0, 255, 0), 2)
        cv2.circle(img, object_uv, 100, (0, 255, 0), 2)
        cv2.imshow("img", img)
        # cv2.imshow("detect_ret_img", detect_ret_img)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

    save_points(detect_points, 1000)


if __name__ == "__main__":
    main()

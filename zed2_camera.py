#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/7/12 15:10
# @Author      :weiz
# @ProjectName :robotVision2
# @File        :zed2_camera.py
# @Description :Encapsulation the zed2 camera
import cv2
import os
import time
import pyzed.sl as zed
import shutil
import math
import numpy as np


class Zed2Camera(object):
    def __init__(self, resolution=None, depth_mode=None):
        """
        initialization of zed2 camera
        :param resolution:resolution of zed2 camera
        :param depth_mode:depth mode of zed2 camera
        """
        self.camera = zed.Camera()
        self.input_type = zed.InputType()
        self.camera_init = zed.InitParameters(input_t=self.input_type, camera_image_flip=zed.FLIP_MODE.OFF)

        # set resolution
        if resolution == "HD2K":
            self.camera_init.camera_resolution = zed.RESOLUTION.HD2K
        elif resolution == "HD1080":
            self.camera_init.camera_resolution = zed.RESOLUTION.HD1080
        else:
            self.camera_init.camera_resolution = zed.RESOLUTION.HD720

        # Set the mode of getting depth information
        if depth_mode == "PERFORMANCE":
            self.camera_init.depth_mode = zed.DEPTH_MODE.PERFORMANCE
        elif depth_mode == "QUALITY":
            self.camera_init.depth_mode = zed.DEPTH_MODE.QUALITY
        else:
            self.camera_init.depth_mode = zed.DEPTH_MODE.ULTRA
        self.camera_init.coordinate_units = zed.UNIT.METER  # millimeter or meter
        self.camera_init.depth_minimum_distance = 0.1

        self.calibration_params = self.camera.get_camera_information().calibration_parameters

        err = self.camera.open(self.camera_init)
        if err != zed.ERROR_CODE.SUCCESS:
            print(repr(err))
            self.camera.close()
            exit(1)

        self.image_size = self.camera.get_camera_information().camera_resolution
        self.image_zed = zed.Mat(self.image_size.width, self.image_size.height, zed.MAT_TYPE.U8_C4)

        # Set runtime parameters after opening the camera
        self.runtime = zed.RuntimeParameters()
        self.runtime.sensing_mode = zed.SENSING_MODE.STANDARD

    def get_ipm_dpv(self, flag=None):
        """
        get the internal parameter matrix and distortion parameter vector of the camera
        :param flag: left or right
        :return:
        """
        if flag == "right":
            fx = self.calibration_params.left_cam.fx
            fy = self.calibration_params.left_cam.fy
            cx = self.calibration_params.left_cam.cx
            cy = self.calibration_params.left_cam.cy
            k1 = self.calibration_params.left_cam.disto[0]
            k2 = self.calibration_params.left_cam.disto[1]
            k3 = self.calibration_params.left_cam.disto[2]
            p1 = self.calibration_params.left_cam.disto[3]
            p2 = self.calibration_params.left_cam.disto[4]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.array([k1, k2, p1, p2, k3])
        else:
            # fx = 1058.28
            # fy = 1057.62
            # cx = 988.82
            # cy = 524.796
            # k1 = -0.0427511
            # k2 = 0.0112919
            # k3 = -0.00517873
            # p1 = 0.000248792
            # p2 = 0.000565185
            fx = 529.14
            fy = 528.81
            cx = 652.91
            cy = 350.898
            k1 = -0.0427511
            k2 = 0.0112919
            k3 = -0.00517873
            p1 = 0.000248792
            p2 = 0.000565185

            fx = 534.18628321
            fy = 539.11107065
            cx = 651.18674139
            cy = 353.39439338
            k1 = -0.04556683
            k2 = 0.08534247
            k3 = 0.00170659
            p1 = 0.00031686
            p2 = -0.04503881

            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.array([k1, k2, p1, p2, k3])

    def data_stream_running(self):
        """
        obtain camera data stream
        :return:
        """
        image_zed_left = zed.Mat()
        image_zed_right = zed.Mat()
        depth_image = zed.Mat(self.image_size.width, self.image_size.height, zed.MAT_TYPE.U8_C4)
        self.point_cloud = zed.Mat()
        self.depth_map = zed.Mat()

        while True:
            self.camera.grab(self.runtime)

            # left image
            self.camera.retrieve_image(image_zed_left, zed.VIEW.LEFT)
            image_cv_left = image_zed_left.get_data()

            # right image
            self.camera.retrieve_image(image_zed_right, zed.VIEW.RIGHT)
            image_cv_right = image_zed_right.get_data()

            # depth information and point cloud
            self.camera.retrieve_image(depth_image, zed.VIEW.DEPTH, zed.MEM.CPU, self.image_size)
            self.camera.retrieve_measure(self.point_cloud, zed.MEASURE.XYZRGBA, zed.MEM.CPU, self.image_size)
            self.camera.retrieve_measure(self.depth_map, zed.MEASURE.DEPTH, zed.MEM.CPU, self.image_size)

            self.image_cv_left = cv2.cvtColor(image_cv_left, 1)
            self.image_cv_right = cv2.cvtColor(image_cv_right, 1)
            self.image_depth = depth_image.get_data()

            yield

    def show_img(self):
        """
        show image
        :return:
        """
        while True:
            next(self.data_stream_running())

            cv2.imshow("left", self.image_cv_left)
            cv2.imshow("right", self.image_cv_right)
            cv2.imshow("depth", self.image_depth)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                break

    def get_opencv_format(self, flag=None):
        """
        get image of opencv format
        :param flag: right or left
        :return:
        """
        next(self.data_stream_running())

        if flag == "right":
            return self.image_cv_right
        else:
            return self.image_cv_left

    def get_depth_value(self, x, y):
        """
        get the depth value of the point (x, y) on the image
        :param x:
        :param y:
        :return:
        """
        next(self.data_stream_running())
        _, value = self.depth_map.get_value(y, x)  # bug of ZED SDK3.5.0,the later version should be fixed
        # print(self.depth_map.get_width(), self.depth_map.get_height())
        return value

    def get_point_cloud(self, x, y):
        """
        get point cloud of the point (x, y) on the image
        :param x:
        :param y:
        :return: [x, y, z, color],distance from the left camera
        """
        next(self.data_stream_running())
        _, point3D = self.point_cloud.get_value(x, y)
        distance = math.sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1] + point3D[2] * point3D[2])
        return point3D, distance

    def save_img_key(self, arm_robot, save_path=None):
        """
        save image according to key information and get posture of robotic arm
        :param arm_robot: object of ur robot or agile robot
        :param save_path:
        :return:
        """
        if save_path == None:
            save_path = "./zed2_images_key"
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
        else:
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
        os.mkdir(save_path)

        robotic_arm_posture = []
        num_left = 1
        num_right = 1
        while True:
            next(self.data_stream_running())

            cv2.imshow("left", self.image_cv_left)
            cv2.imshow("right", self.image_cv_right)
            cv2.imshow("depth", self.image_depth)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('l'):
                tmp_path = os.path.join(save_path, "L{:0>3d}.png".format(num_left))
                cv2.imwrite(tmp_path, self.image_cv_left)
                print(tmp_path)
                num_left = num_left + 1

                robotic_arm_posture.append(arm_robot.get_robotic_arm_posture())
            if key & 0xFF == ord('r'):
                tmp_path = os.path.join(save_path, "R{:0>3d}.png".format(num_right))
                cv2.imwrite(tmp_path, self.image_cv_right)
                print(tmp_path)
                num_right = num_right + 1

                robotic_arm_posture.append(arm_robot.get_robotic_arm_posture())
            if key & 0xFF == 27:
                cv2.destroyAllWindows()
                break

        return robotic_arm_posture

    def save_img(self, img, save_path=None, img_name=None):
        """
        save image
        :param img:
        :param save_path:
        :param img_name:
        :return:
        """
        if save_path == None:
            save_path = "./zed2_images"
        if img_name == None:
            img_name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = os.path.join(save_path, img_name + ".png")

        cv2.imwrite(save_path, img)


def main():
    cam = Zed2Camera()
    cam.show_img()

if __name__ == "__main__":
    main()
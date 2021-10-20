#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/7/12 9:12
# @Author      :weiz
# @ProjectName :robotVision2
# @File        :calibrate.py
# @Description :Camera calibration, including camera calibration and hand-eye calibration.
import cv2
import numpy as np
import sys
import os
from common import read_line


class Calibrate(object):
    def __init__(self, cb_size, side_length):
        """
        initialization of class
        :param cb_size:    size(row, column) of calibration board. eg:(6, 8)
        :param side_length:side length of the calibration board grid. eg:0.035m
        """
        (self.rows, self.cols) = cb_size
        self.side_length = side_length

        self.ipm = np.zeros((3, 3))  # camera internal parameter matrix. [3*3]
        self.dpv = 0  # camera distortion parameter vector. [1*5]

        self.R_camera2gripper = 0  # Eyes in hands: the rotation matrix from the camera to the end of the robotic arm
        self.t_camera2gripper = 0  # Eyes in hands: the translation vector from the camera to the end of the robotic arm

        self.R_camera2base = 0  # Eyes to hands: the rotation matrix from the camera to the base of the robotic arm
        self.t_camera2base = 0  # Eyes to hands: the translation vector from the camera to the base of the robotic arm

    def find_corners(self, img, is_show=False):
        """
        find corners on image
        :param img:
        :param is_show:
        :return:
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray", gray)
        # cv2.waitKey(0)

        # Find the corner
        ret, corners = cv2.findChessboardCorners(gray, (self.cols, self.rows), None)
        if not ret:
            return ret, corners

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # sub-pixel corner detection
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if is_show:
            showRet = cv2.drawChessboardCorners(img, (self.cols, self.rows), corners2, ret)
            cv2.imshow('corner', showRet)
            cv2.waitKey(0)

        return ret, corners2

    def calibrate_camera(self, path_img_folder):
        """
        calibrate of camera
        :param path_img_folder: folder containing pictures
        :return:
        """
        cb_coordinate = np.zeros((self.cols * self.rows, 3), np.float32)
        cb_coordinate[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)  # coordinates of calibration board

        points_3D = []  # 3D point in real world space
        points_2D = []  # 2D point in image plane

        img_name_list = os.listdir(path_img_folder)
        for img_name in img_name_list:
            img_path = os.path.join(path_img_folder, img_name)
            img = cv2.imread(img_path)

            ret_corners, corners2 = self.find_corners(img, True)

            if ret_corners:
                points_3D.append(cb_coordinate)
                points_2D.append(corners2)

        cv2.destroyAllWindows()

        # R:rotation matrix; t:translation vector
        ret, self.ipm, self.dpv, R, t = cv2.calibrateCamera(points_3D, points_2D,
                                                            img.shape[::-1][1:], None, None)

        # Calculate the average error
        total_error = 0
        for i in range(len(points_3D)):
            img_points, _ = cv2.projectPoints(points_3D[i], R[i], t[i], self.ipm, self.dpv)
            error = cv2.norm(points_2D[i], img_points, cv2.NORM_L2) / len(img_points)
            total_error += error
        print("Average Error：", total_error / len(points_3D))

        return self.ipm, self.dpv, R, t

    def calibrate_img(self, img):
        """
        Picture correction
        :param img:
        :return:
        """
        if self.ipm.any() == 0:
            print("Please calibrate the camera first!")
            return

        dest_img = cv2.undistort(img, self.ipm, self.dpv)
        cv2.imshow('source', img)
        cv2.imshow('dest', dest_img)
        cv2.waitKey(0)

    def get_gripper2base_R_t(self, gripper_posture, image_not_corner_ind):
        """
        Calculate the rotation matrix and translation matrix from the end of the robot arm to the base of the robot arm
        :param gripper_posture: [[x, y, z, rx, ry, rz]...],eg:[[0.666,0.666,0.666,1.2,1.2,1.2]...];unit:meter, rad
        :param image_not_corner_ind:
        :return:
        """
        for value in image_not_corner_ind[::-1]:  # Flashback delete
            gripper_posture.pop(value)
            print("Delete the pose data of the robotic arm with index {}".format(value))

        R_gripper2base_array = []
        t_gripper2base_array = []
        for arr in gripper_posture:
            R_target2cam = np.zeros((3, 3))
            cv2.Rodrigues(np.array([arr[3], arr[4], arr[5]]), R_target2cam)
            R_gripper2base_array.append(R_target2cam)

            t_gripper2base_array.append(np.array([arr[0], arr[1], arr[2]]))

            # print("rotation matrix:", R_target2cam)
            # print("translation vector:", np.array(np.array([arr[0], arr[1], arr[2]]))

        return R_gripper2base_array, t_gripper2base_array

    def get_target2camera_R_t(self, ipm, dpv, path_img_folder):
        """
        Calculate the rotation matrix and translation matrix from the calibration board to the camera.
        :param ipm:
        :param dpv:
        :param path_img_folder: folder containing pictures
        :return:
        """
        cb_coordinate = np.zeros((self.cols * self.rows, 3), np.float32)
        # coordinates of calibration board
        cb_coordinate[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2) * self.side_length

        R_target2cam_array = []
        t_target2cam_array = []

        image_name_list = []
        name_list_all = os.listdir(path_img_folder)
        for file_name in name_list_all:
            if file_name.split('.')[-1] not in ["jpg", "png", "jpeg"]:
                print("The {} is not image!".format(os.path.join(path_img_folder, file_name)))
                continue
            image_name_list.append(file_name)
        image_name_list.sort()
        image_not_corner_ind = []  # Record image where no corner points are detected
        images_delete = []
        for ind, img_name in enumerate(image_name_list):
            img_path = os.path.join(path_img_folder, img_name)
            img = cv2.imread(img_path)

            ret_corners, corners2 = self.find_corners(img, True)
            if not ret_corners:
                print("{} no corners detected!Its index is:{}".format(img_name, ind))
                image_not_corner_ind.append(ind)
                images_delete.append(img_path)
                continue

            ret, R_target2cam_vec, t_target2cam = cv2.solvePnP(cb_coordinate, corners2, ipm, dpv)
            if ret:
                R_target2cam = np.zeros((3, 3))
                cv2.Rodrigues(R_target2cam_vec, R_target2cam)
                # print("rotation matrix:", R_target2cam)
                # print("translation vector:", t_target2cam)
                R_target2cam_array.append(R_target2cam)
                t_target2cam_array.append(t_target2cam)

        if len(images_delete) > (len(image_name_list) / 2):
            print("The parameter settings may be wrong!")
            sys.exit()
        else:
            for image_delete in images_delete:
                os.remove(image_delete)
                print("The {} is detected!".format(image_delete))

        return R_target2cam_array, t_target2cam_array, image_not_corner_ind

    def calibrate_hand_in_eye(self, gripper_posture, ipm, dpv, path_img_folder):
        """
        Realize eye-on-hand hand-eye calibration
        :param gripper_posture: [[x, y, z, rx, ry, rz]...],eg:[[0.666,0.666,0.666,1.2,1.2,1.2]...];unit:meter, rad
        :param ipm:
        :param dpv:
        :param path_img_folder: folder containing pictures
        :return:
        """
        # Obtain the rotation matrix and translation matrix from the calibration board to the camera
        R_target2cam_array, t_target2cam_array, image_not_corner_ind = self.get_target2camera_R_t(ipm, dpv,
                                                                                                  path_img_folder)

        # Obtain the rotation matrix and translation matrix from the end of the robot arm to the base of the robot arm
        R_gripper2base_array, t_gripper2base_array = self.get_gripper2base_R_t(gripper_posture, image_not_corner_ind)

        self.R_camera2gripper, self.t_camera2gripper = cv2.calibrateHandEye(R_gripper2base_array, t_gripper2base_array,
                                                                      R_target2cam_array, t_target2cam_array,
                                                                      method=cv2.CALIB_HAND_EYE_PARK)

        return self.R_camera2gripper, self.t_camera2gripper

    def calibrate_hand_to_eye(self, gripper_posture, ipm, dpv, path_img_folder):
        """
        Realize eye-to-hand hand-eye calibration
        :param gripper_posture: [[x, y, z, rx, ry, rz]...],eg:[[0.666,0.666,0.666,1.2,1.2,1.2]...];unit:meter, rad
        :param ipm:
        :param dpv:
        :param path_img_folder: folder containing pictures
        :return:
        """
        # Obtain the rotation matrix and translation matrix from the calibration board to the camera
        R_target2cam_array, t_target2cam_array, image_not_corner_ind = self.get_target2camera_R_t(ipm, dpv,
                                                                                                  path_img_folder)

        # Obtain the rotation matrix and translation matrix from the end of the robot arm to the base of the robot arm
        R_gripper2base_array, t_gripper2base_array = self.get_gripper2base_R_t(gripper_posture, image_not_corner_ind)
        # Turn into a rotation matrix and translation matrix from the base of the robot arm to the end of the robot arm
        R_base2gripper_array = []
        t_base2gripper_array = []
        for mat in R_gripper2base_array:
            R_base2gripper = np.linalg.inv(np.array(mat))
            R_base2gripper_array.append(R_base2gripper)
        for i, mat in enumerate(t_gripper2base_array):
            t_base2gripper = -np.dot(R_base2gripper_array[i], mat)
            t_base2gripper_array.append(t_base2gripper)

        self.R_camera2base, self.t_camera2base = cv2.calibrateHandEye(R_base2gripper_array, t_base2gripper_array,
                                                                      R_target2cam_array, t_target2cam_array,
                                                                      method=cv2.CALIB_HAND_EYE_PARK)

        return self.R_camera2base, self.t_camera2base

    def calibrate_offline(self, images_path, posture_name, ipm, dpv, mode=None):
        """
        calibrate of offline
        :param images_path:
        :param posture_name:
        :param ipm:camera internal parameter matrix. [3*3]
        :param dpv:camera distortion parameter vector. [1*5]
        :param mode: calibrate hand in/to eye. mode = ["in", "to"]
        :return:
        """
        gripper_posture_str = read_line(posture_name)
        gripper_posture = []
        for line in gripper_posture_str:
            if len(line) == 0:
                continue
            if line[-1] == ',':
                line = line[:-1]

            line = line.replace('[', '').replace(']', '').replace(' ', '')
            if line.count(',') == 5:
                x, y, z, rx, ry, rz = line.split(',')
                gripper_posture.append([float(x), float(y), float(z), float(rx), float(ry), float(rz)])
                print([float(x), float(y), float(z), float(rx), float(ry), float(rz)])

        if mode == "to":  # calibrate hand to eye
            R_camera2base, t_camera2base = self.calibrate_hand_to_eye(gripper_posture, ipm, dpv, images_path)
            return R_camera2base, t_camera2base
        else:
            R_camera2gripper, t_camera2gripper = self.calibrate_hand_in_eye(gripper_posture, ipm, dpv, images_path)
            return R_camera2gripper, t_camera2gripper

def main():
    cali_camera = Calibrate((6, 8), 0.035)
    # ipm, dpv, R, t = cali_camera.calibrate_camera("./abc11")
    # print(ipm)
    # print(dpv)
    # img = cv2.imread("./abc11/L001.png")
    # cali_camera.calibrate_img(img)
    ipm = np.array([[534.18628321, 0, 651.18674139], [0, 539.11107065, 353.39439338], [0, 0, 1]])
    dpv = np.array([-0.04556683, 0.08534247, 0.00031686, -0.04503881, 0.00170659])
    R, t = cali_camera.calibrate_offline("./test", "./2021_08_20_13_22_35.txt", ipm, dpv)
    print(R)
    print(t)


if __name__ == "__main__":
    main()
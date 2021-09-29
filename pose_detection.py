#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/7/15 14:12
# @Author      :weiz
# @ProjectName :robotVision2
# @File        :pose_detection.py
# @Description :object pose detection
import cv2
import copy
import numpy as np
import math


class PoseDetection(object):
    def __init__(self, cb_size):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.cb_size = cb_size

    @staticmethod
    def draw_marker(ids, side_pixels):
        """
        draw a canonical marker image
        :param ids:id of mark,eg:[1,12,36,45,32]
        :param side_pixels:size of the image in pixels,eg:256
        :return:
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        for id in ids:
            img = cv2.aruco.drawMarker(aruco_dict, id, side_pixels)
            img_name = "aruco_" + str(id) + ".png"
            cv2.imwrite(img_name, img)

    def get_aruco_dict(self):
        """
        :return:
        """
        return self.aruco_dict

    def draw_result(self, img, ipm, dpv, corners, R_vectors, t_vectors):
        """

        :param img:
        :param ipm:internal parameter matrix
        :param dpv:distortion parameter vector
        :param corners:
        :param R_vectors:[[[rx, ry, rz],[...]]]
        :param t_vectors:[[[rx, ry, rz],[...]]]
        :return:
        """
        cv2.line(img, (10, 20), (50, 20), (0, 0, 255), 2)
        cv2.putText(img, "X", (55, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.line(img, (10, 40), (50, 40), (0, 255, 0), 2)
        cv2.putText(img, "Y", (55, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.line(img, (10, 60), (50, 60), (255, 0, 0), 2)
        cv2.putText(img, "Z", (55, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        for i in range(R_vectors.shape[0]):
            cv2.aruco.drawAxis(img, ipm, dpv, R_vectors[i, :, :], t_vectors[i, :, :], 0.03)
            cv2.aruco.drawDetectedMarkers(img, corners)

        return img

    def get_all_marker_rotation_vector(self, img, ipm, dpv, marker_length):
        """
        Return the posture vectors of all markers in the picture
        :param img:input image
        :param ipm:internal parameter matrix
        :param dpv:distortion parameter vector
        :param marker_length:length of marker
        :return:
        """
        img = copy.deepcopy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # lists of ids and the corners beloning to each id
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict,
                                                  parameters=cv2.aruco.DetectorParameters_create())
        if ids is not None:
            self.R_vectors, self.t_vectors, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, ipm, dpv)
            img = self.draw_result(img, ipm, dpv, corners, self.R_vectors, self.t_vectors)
        else:
            self.R_vectors = [[]]
            self.t_vectors = [[]]

        return self.R_vectors, self.t_vectors, img, corners, ids

    def get_id_marker_rotation_vector(self, img, ipm, dpv, marker_length, id):
        """
        Return the posture vectors of the id markers in the picture
        :param img:
        :param ipm:
        :param dpv:
        :param marker_length:
        :param id:
        :return:
        """
        R_vectors, t_vectors, _, corners, ids = self.get_all_marker_rotation_vector(img, ipm, dpv, marker_length)

        corner = []
        R_vector = []
        t_vector = []
        if ids is not None:
            for ind, val in enumerate(ids):
                if id == val:
                    corner.append(corners[ind])
                    R_vector.append(R_vectors[ind])
                    t_vector.append(t_vectors[ind])
            img = self.draw_result(img, ipm, dpv,
                                   np.array(corner), np.array(R_vector),
                                   np.array(t_vector))

            euler_angles = self.transfer_rotation_vector2euler_angles(R_vectors[ind, :, :])
            return np.squeeze(np.array(R_vector)), np.squeeze(np.array(t_vector)), img, euler_angles
        else:
            return np.array([0, 0, 0]), np.array([0, 0, 0]), img, np.array([0, 0, 0])

    def get_rotation_vector_of_pnp(self, img, ipm, dpv, marker_length, id):
        """

        :param img:
        :param ipm:
        :param dpv:
        :param marker_length:
        :param id:
        :return:
        """
        img = copy.deepcopy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # lists of ids and the corners beloning to each id
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict,
                                                  parameters=cv2.aruco.DetectorParameters_create())

        if ids is not None:
            for ind, val in enumerate(ids):
                if id == val:
                    corners_id = np.squeeze(corners[ind])
                    cv2.putText(img, "0", (int(corners_id[0][0]), int(corners_id[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)
                    cv2.putText(img, "1", (int(corners_id[1][0]), int(corners_id[1][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)
                    cv2.putText(img, "2", (int(corners_id[2][0]), int(corners_id[2][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)
                    cv2.putText(img, "3", (int(corners_id[3][0]), int(corners_id[3][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)
                    # cv2.imshow("show points", img)
                    # Construct a world coordinate system
                    points_3D = [[0, 0, 0], [marker_length, 0, 0], [marker_length, marker_length, 0],
                                 [0, marker_length, 0]]
                    ret, R_target2cam_vec, t_target2cam = cv2.solvePnP(np.array(points_3D), corners_id, ipm, dpv)
                    if ret:
                        euler_angles = self.transfer_rotation_vector2euler_angles(R_target2cam_vec)
                        return R_target2cam_vec, t_target2cam, img, euler_angles
                    else:
                        return np.array([0, 0, 0]), np.array([0, 0, 0]), img, np.array([0, 0, 0])
        return np.array([0, 0, 0]), np.array([0, 0, 0]), img, np.array([0, 0, 0])

    def transfer_rotation_vector2euler_angles(self, rvecs):
        """
        Rotation vector to Euler angle
        :param rvecs:
        :return:
        """
        R = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(rvecs, R)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        x = x * 180.0 / 3.141592653589793
        y = y * 180.0 / 3.141592653589793
        z = z * 180.0 / 3.141592653589793
        return [x, y, z]

    def find_corners(self, img, is_show=False, show_time=0):
        """

        :param img:
        :param is_show:
        :param show_time:
        :return:
        """
        (rows, cols) = self.cb_size
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the corner
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # sub-pixel corner detection
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners2
        else:
            return False, []

    def cb_pose_detection(self, img, ipm, dpv):
        (rows, cols) = self.cb_size

        cb_3D = np.zeros((cols * rows, 3), np.float32)
        cb_3D[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  # coordinates of calibration board

        ret, corners2 = self.find_corners(img)
        if ret:
            ret, R_vector, t = cv2.solvePnP(cb_3D, corners2, ipm, dpv)
            euler_angles = self.transfer_rotation_vector2euler_angles(R_vector)
            return True, R_vector, euler_angles
        else:
            return False, [], []



def main():
    pd = PoseDetection((6, 8))
    cap = cv2.VideoCapture(0)

    ipm = np.array([[2946.48, 0, 1980.53],
                    [0, 2945.41, 1129.25],
                    [0, 0, 1]])
    dpv = np.array([0.226317, -1.21478, 0.00170689, -0.000334551, 1.9892])

    while True:
        ret, frame = cap.read()
        # R_vectors, t_vectors, img, corners, ids = pd.get_all_marker_rotation_vector(frame, ipm, dpv, 0.04)
        # print(R_vectors)

        # R_vector, t_vector, img, euler_angles = pd.get_id_marker_rotation_vector(frame, ipm, dpv, 0.04, 6)
        # print(euler_angles)

        # _, R, euler_angles = pd.cb_pose_detection(frame, ipm, dpv)
        # print(euler_angles)

        R_vector, t_vector, img, euler_angles = pd.get_rotation_vector_of_pnp(frame, ipm, dpv, 0.135, 6)
        print(euler_angles)

        cv2.imshow("frame", frame)
        cv2.imshow("result", img)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
    # PoseDetection.draw_marker([1,2,3,4,5,6], 512)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/8/19 10:35
# @Author      :weiz
# @ProjectName :agile_robot
# @File        :agile_robot.py
# @Description :
# Copyright (C) 2021-2025 Jiangxi Institute Of Intelligent Industry Technology Innovation
from agile_robot_folder.DianaApi import *
import time


class AgileRobot:
    def __init__(self, host, port=None):
        """
        initialization of agile robot
        :param host: ip of host
        :param port:
        """
        self.host = host
        if port is None or port == '':
            self.port = ''
        else:
            self.port = port
        self.speed = 0.1
        self.acceleration = 0.2

        # link agile robot
        initSrv((self.host, 0, 0, 0))
        releaseBrake()

    def __del__(self):
        """
        Destructor, release resources
        :return:
        """
        holdBrake()
        destroySrv()

    def send_order_digital(self, n, flag):
        """

        :param n:
        :param flag:
        :return:
        """
        if n == 0:
            do_name = 'do0'
            writeDO('board', do_name, 2, flag)
        else:
            print("send_order_digital:input error!")
        time.sleep(1)

    def send_order_gripper(self, pose, speed=None, acceleration=None):
        """

        :param pose:
        :param speed:
        :param acceleration:
        :return:
        """
        if speed is None:
            speed_input = self.speed
        else:
            speed_input = speed
        if acceleration is None:
            acceleration_input = self.acceleration
        else:
            acceleration_input = acceleration
        # print("pose = ",pose)
        # print("speed = ",speed_input)
        # print("acc = ",acceleration_input)
        self.moveJToPose(pose, speed_input, acceleration_input)
        # time.sleep(2)
        
        # print("movej send ok!")
        error = 0.005
        while True:
            cur_position = self.get_robotic_arm_posture()
            if (abs(pose[0] - cur_position[0]) < error) and (abs(pose[1] - cur_position[1]) < error) and (abs(pose[2] - cur_position[2]) < error):
                break
            time.sleep(1)

    def get_robotic_arm_posture(self):
        """

        :return:
        """
        pose_arm = [0.0] * 6
        getTcpPos(pose_arm)
        return pose_arm

    def moveJToPose(self, pose, speed, acceleration):
        """

        :param pose:
        :param speed:
        :param acceleration:
        :return:
        """
        stop()
        moveJToPose(pose, speed, acceleration)

    def connect_robotic_arm(self):
        """

        :return:
        """
        while True:
            pose_arm = self.get_robotic_arm_posture()
            print(pose_arm)

            input_data = input("input data[x, y, z, rx, ry, rz] or 'exit':")
            if input_data == "exit":
                break

            if input_data == '':
                continue
            else:
                x1, x2, x3, x4, x5, x6 = input_data.replace(" ", "").split(',')
                pose_input = [float(x1), float(x2), float(x3), float(x4), float(x5), float(x6)]
                self.moveJToPose(pose_input, self.speed, self.acceleration)


host = "192.168.10.75"
port = ""
# [0.392, 0.371, 0.433, 0.002, 0.011, -0.653]
# 0.42190311302382993, 0.40514241376478327, 0.4343071485036608, -0.21795725565454088, 0.6067600673883295, 0.4139117116153281
if __name__ == "__main__":
    agile = AgileRobot(host, port)

    agile.connect_robotic_arm()
    # agile.send_order_digital(0, 0)
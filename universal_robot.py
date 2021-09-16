#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/8/4 13:33
# @Author      :weiz
# @ProjectName :robotVision2
# @File        :universal_robot.py
# @Description :Control the UR robotic arm and obtain data
import socket
import struct


class UR(object):
    def __init__(self, host, port, buf):
        """
        ur10 initialization
        :param host:
        :param port:
        :param buf:
        """
        self.host = host
        self.port = port
        self.buf = buf

    def data_analysis(self, data):
        """
        Analyze Universal Robots data. Note that this is related to the version and model of the machine
        :param data:
        :return:
        """
        # f = open('data.txt', 'a')
        # f.write('\n'+str(data))
        # f.close()

        dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d',
               'I target': '6d',
               'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
               'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
               'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
               'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
               'Tool Accelerometer values': '3d', 'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd',
               'SoftwareOnly': 'd', 'softwareOnly2': 'd', 'V main': 'd', 'V robot': 'd', 'I robot': 'd',
               'V actual': '6d',
               'Digital outputs': 'd', 'Program state': 'd', 'Elbow position': '3d', 'Elbow velocity': '3d'}
        names = []
        ii = range(len(dic))
        for key, i in zip(dic, ii):
            fmtsize = struct.calcsize(dic[key])
            data1, data = data[0:fmtsize], data[fmtsize:]
            fmt = "!" + dic[key]
            names.append(struct.unpack(fmt, data1))
            if key == "Tool vector target":
                tcp_data = (float(struct.unpack(fmt, data1)[0]), float(struct.unpack(fmt, data1)[1]),
                            float(struct.unpack(fmt, data1)[2]), float(struct.unpack(fmt, data1)[3]),
                            float(struct.unpack(fmt, data1)[4]), float(struct.unpack(fmt, data1)[5]))
                dic[key] = dic[key], tcp_data
            else:
                dic[key] = dic[key], struct.unpack(fmt, data1)

        return dic

    def build_list(self, data):
        """
        Build order of format of universal robots
        :param data:
        :return:
        """
        if data == '':
            return ''
        x1, x2, x3, x4, x5, x6 = data.replace(" ", "").split(',')

        return [float(x1), float(x2), float(x3), float(x4), float(x5), float(x6)]

    def send_order_digital(self, n, flag):
        """
        send order to digital out
        :param n:index of digital out. 0<= n <= 7
        :param flag:Boolean value
        :return:
        """
        addr = (self.host, self.port)
        tcp_sock = socket.socket()
        tcp_sock.connect(addr)
        # set_standard_digital_out(0, True)
        # set_tool_digital_out(0, True)
        # set_configurable_digital_out(0, True)

        if flag in [1, "true", "True", '1']:
            flag = "True"
        elif flag in [0, "False", "false", '0']:
            flag = "False"
        else:
            print("[send_order_digital()]:input error!")
            return

        order = '''
                def endMovement():
                    set_standard_digital_out({}, {})
                '''.format(n, flag)

        tcp_sock.send(order.encode("utf-8"))
        tcp_sock.close()

    def send_order_gripper(self, posture):
        """
        send order to gripper
        :param posture:
        :return:
        """
        addr = (self.host, self.port)
        tcp_sock = socket.socket()
        tcp_sock.connect(addr)

        order = '''
                def endMovement():
                    movej(p[{},{},{},{},{},{}])
                '''.format(float(posture[0]), float(posture[1]), float(posture[2]),
                           float(posture[3]), float(posture[4]), float(posture[5]))
        tcp_sock.send(order.encode("utf-8"))
        tcp_sock.close()

    def get_robotic_arm_posture(self):
        """
        get robotic arm data
        :return: [x, y, x, rx, ry, rz]
        """
        addr = (self.host, self.port)
        tcp_sock = socket.socket()
        tcp_sock.connect(addr)

        data = tcp_sock.recv(buf)
        data_dic = self.data_analysis(data)
        tcp_sock.close()

        return list(data_dic["Tool vector target"][1])

    def connect_robotic_arm(self):
        """
        connect universal robots
        :return:
        """
        while True:
            input_data = input("input data[x, y, z, rx, ry, rz] or 'exit':")
            if input_data == "exit":
                break
            posture = self.build_list(input_data)

            addr = (self.host, self.port)
            tcp_sock = socket.socket()
            tcp_sock.connect(addr)
            data = tcp_sock.recv(self.buf)
            data_dic = self.data_analysis(data)
            print("Robotic Arm posture:", data_dic["Tool vector target"][1])
            if posture == '':
                continue
            tcp_sock.close()

            self.send_order_gripper(posture)


port = 30003
host = "192.168.2.128"  # 192.168.10.167   192.168.207.128
buf = 1140
# eg:0.59453, 0.43099, 0.81997, 4.209, 1.270, -0.384
#    0.92448, 0.50774, 0.62857, 3.618, -1.531, 0.824
def main():
    ur10 = UR(host, port, buf)

    # ur10.send_order_digital(0, 0)
    # ur10.send_order_gripper([0.59453, 0.43099, 0.81997, 4.209, 1.270, -0.384])
    # print(ur10.get_robotic_arm_posture())
    ur10.connect_robotic_arm()


if __name__ == "__main__":
    main()
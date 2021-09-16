#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        :2021/7/20 14:08
# @Author      :chentao
# @ProjectName :yolov5_jetson
# @File        :main.py
# @Description :
import sys
sys.path.append('/home/sky/Project/yolov5-jetson/build')
import cv2
import math
import _thread
import numpy as np
import yolov5_module
from universal_robot import UR
from   calibrate import Calibrate
from   zed2_camera import Zed2Camera
from   common import *


def guide_stop_falg():
     while (not stop_flag):
        c = input()
        if(c == 'q'):
            stop_flag = True
       
def transfer_rotation_vector2euler_angles(R):
    """
    Rotation vector to Euler angle
    :param rvecs:
    :return:
    """
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
    return x, y, z 


def calibrate(cam):
    """
    calibrate
    :param cam:
    :return:
    """
    # Collect hand-eye calibration images and the pose of the robotic arm
    # cam.show_img()
    robotic_arm_posture = cam.save_img_key(host, port, buf, hand_eye_img_path)

    # get R t
    calibrate = Calibrate(cb_size, side_length)
    ipm, dpv = cam.get_ipm_dpv()
    R_camera2gripper, t_camera2gripper = calibrate.calibrate_hand_in_eye(robotic_arm_posture, ipm, dpv,
                                                                         hand_eye_img_path)

    return R_camera2gripper, t_camera2gripper


def guide_robotic_arm(object_pixel_coord, ur, cam, pose_detection, R_camera2gripper, t_camera2gripper):
    """
    Guide the robotic arm to grab
    :param object_pixel_coord:
    :param ur:
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
    gripper_posture = ur.get_robotic_arm_posture()

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
    # ur.send_order_gripper([base_arm_3D_object[0], base_arm_3D_object[1], base_arm_3D_object[2],
    #                       R_vector_posture_base[0], R_vector_posture_base[1], R_vector_posture_base[2]])


def get_origin_target_rect(img_w, img_h, dect_w, dect_h, bbox):
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

def read_line_file(path):
    """
    Read file by line
    :param path:
    :return:
    """
    txtLines = []
    file = open(path)
    for line in file:
        line = line.strip('\n')
        txtLines.append(line)
    file.close()
    return txtLines

def judge_tcp_arrive(cur_pos, dst_pos, thr_arrive):
    """
    judge whether tcp  arrive the target 
    :param cur_pos: current pose [x,y,z,rx,ry,rz]
    :param dst_pos: dst pose [x,y,z,rx,ry,rz]
    :param thr_arrive: the arrive range thr
    :return: Bool whether arrive
    """
    flag = False

    if ((abs(cur_pos[0] - dst_pos[0]) < thr_arrive) and (abs(cur_pos[1] - dst_pos[1]) < thr_arrive) and\
        (abs(cur_pos[2] - dst_pos[2]) < thr_arrive)) :
        flag = True
    
    return flag


cb_size = (6, 9)
side_length = 0.035
hand_eye_img_path = "./data/calib_img"

port = 30003
host = "192.168.10.138"  # 192.168.10.167   192.168.207.128
buf = 1140


def main():
    stop_flag = False
    engine_path  = '/home/sky/Project/yolov5-jetson/build/yolov5s.engine'
    cam = Zed2Camera()
    ur = UR(host, port, buf)
    # R_camera2gripper, t_camera2gripper = calibrate(cam)

    # print(R_camera2gripper)
    # print(t_camera2gripper)
    
    R_camera2gripper =[[ 0.0043702,   0.99923192,  0.03894191],
                        [-0.99998781,  0.00445644, -0.00212792],
                        [-0.00229982, -0.03893214,  0.99923921]]
    t_camera2gripper = [-0.07298794, 0.07396859, -0.00273542]


    yolov5_module.init_inference(engine_path)
    labels = read_line_file("test.names")

    #test
    # depth_pre  = 0.56508183
    # tmp_uv = [738 , 409]
    # guide_robotic_arm(host, port, buf, cam, R_camera2gripper, t_camera2gripper, tmp_uv, depth_pre)
    #test

    # try:
    #     _thread.start_new_thread(detectInputKey, ())
    # except:
    #     print ("Error: can't start thread!\n")

    flag_exit  = 0
    flag_arrive = False
    thr_arrive = 0.005
    orig_pos  = ur.get_robotic_arm_posture(host, port, buf)
    next_pos = [orig_pos[0] + 0.30,  orig_pos[1], 0.15 , orig_pos[3], orig_pos[4], orig_pos[5]]
    
    while (not stop_flag):
        img = cam.get_opencv_format()
        res = yolov5_module.image_inference(img) # target detection
        # print(res[0].bbox)
        # print(res)
        

        #get one object
        for r in res:
            print("class is :", r.classid)
            rect = get_origin_target_rect(cam.image_size.width, cam.image_size.height, 608, 608, r.bbox)
    
            p1 = (int(rect[0]), int(rect[1]))
            p2 = (int(rect[0] + rect[2]), int(rect[1] + rect[3]))
            cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
            cv2.putText(img, labels[int(r.classid)]+str(int(r.classid)), p1, cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)

            if (0 == int(r.classid) and 0 == flag_exit):
                
                print("get stell box object\n")

                #center point
                u = int(round(rect[0] + rect[2] / 2.0))
                v = int(round(rect[1] + rect[3] / 2.0))
                print("get stell object center:",u, v)
                object_u_v = [u, v]
                # tmp_uv = object_u_v
                depth = cam.get_depth_value(object_u_v[1], object_u_v[0]) #compensate for error
                if (math.isnan(depth) or math.isinf(depth) or depth < 0.3 or depth > 2.0):
                    print("depth:", depth)
                    break
                
                # depth_pre = depth
                # tmp_uv = object_u_v
        
                target_pos = guide_robotic_arm(host, port, buf, cam, R_camera2gripper, t_camera2gripper, object_u_v, depth)
                stop_flag = True

                break
            else :
                continue
        
        #show_img
        cv2.imshow("frame", img)

        #calculate
        if cv2.waitKey(1) & 0xFF == 27:
            flag_exit = 1
            break

    if (flag_exit == 0) :
        # guide_robotic_arm(host, port, buf, cam, R_camera2gripper, t_camera2gripper, object_u_v, depth_pre)
        while (not flag_arrive):
            cur_pos    = ur.get_robotic_arm_posture()
            flag_arrive = judge_tcp_arrive(cur_pos, target_pos, thr_arrive)
        #
        ur.send_order_digital(1, 0)
        cv2.waitKey(100)
        ur.send_order_gripper(orig_pos)
        flag_arrive = False
        while (not flag_arrive):
            cur_pos    = ur.get_robotic_arm_posture()
            flag_arrive = judge_tcp_arrive(cur_pos, orig_pos, thr_arrive)
        #
        ur.send_order(next_pos)
        flag_arrive = False
        while (not flag_arrive):
            cur_pos    = ur.get_robotic_arm_posture()
            flag_arrive = judge_tcp_arrive(cur_pos, next_pos, thr_arrive)        
        ur.send_order_digital(0, 0)
        cv2.waitKey(100)
        flag_arrive = False
        ur.send_order_gripper(orig_pos)
        while (not flag_arrive):
            cur_pos    = ur.get_robotic_arm_posture()
            flag_arrive = judge_tcp_arrive(cur_pos, orig_pos, thr_arrive)        
        flag_arrive = False     
        # cv2.waitKey(5000)
    else :
        tmp_target = ur.get_robotic_arm_posture()
        ur.send_order(tmp_target)

    
    yolov5_module.destory_inference()

if __name__ == "__main__":
    main()

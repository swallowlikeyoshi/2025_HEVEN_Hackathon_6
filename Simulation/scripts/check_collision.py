#!/usr/bin/env python3

import rospy
import numpy as np
import time
import cv2
import tkinter as tk
import threading
import os

from math import pi
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
from racecar_simulator.msg import CenterPose, Complete
from parameter_list import Param
from ackermann_msgs.msg import AckermannDrive
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Bool

param = Param()

check_array = np.zeros(shape=360)

for i in range(0, 26):
    check_array[i] = param.REAR_LIDAR / np.cos(i*pi/180)

for i in range(26, 123):
    check_array[i] = param.WIDTH / np.sin(i*pi/180)

for i in range(123, 181):
    check_array[i] = - (param.WHEELBASE - param.REAR_LIDAR) / np.cos(i*pi/180)

for i in range(181, 360):
    check_array[i] = check_array[360-i]

initial_clock = 0
timer_started = False
timer_stamp = 0

# timer_stamp is the time when brain.launch is launched
def timer_starter_callback(msg):
    global timer_started
    global timer_stamp
    if msg.data and not timer_stamp:
        timer_started = True
        timer_stamp = rospy.get_time()

class CheckCollide():
    def __init__(self):
        self.lidar_data = None
        self.initial_clock = rospy.get_time()
        self.collision_count = 0
        # Index of Spawn list
        self.spawn_index = 0
        rospy.Subscriber('scan', LaserScan, self.callback)
        rospy.Subscriber('stop_complete', Complete, self.stop_callback)
        rospy.Subscriber('obs_complete', Complete, self.obs_callback)
        rospy.Subscriber('scurve_complete', Complete, self.scurve_callback)

    def callback(self, data):
        self.lidar_data = np.array(data.ranges)
    
    def stop_callback(self, data):
        if data.complete:
            self.spawn_index = 1
        else:
            self.spawn_index = 0

            # refresh time & collision
            check_col.collision_count = 0
            check_col.initial_clock = rospy.get_time()

            # initialize spawn_index
            check_col.spawn_index = 0

    def obs_callback(self, data):
        if data.complete:
            self.spawn_index = 2
        else:
            self.spawn_index = 0

            # refresh time & collision
            check_col.collision_count = 0
            check_col.initial_clock = rospy.get_time()

            # initialize spawn_index
            check_col.spawn_index = 0

    def scurve_callback(self, data):
        if data.complete:
            self.spawn_index = 3
        else:
            self.spawn_index = 0

            # refresh time & collision
            check_col.collision_count = 0
            check_col.initial_clock = rospy.get_time()

            # initialize spawn_index
            check_col.spawn_index = 0


class CheckEnd():
    def __init__(self):
        self.if_end = False
        self.pose_data = None
        rospy.Subscriber('car_center', CenterPose, self.pose_callback)
        rospy.Subscriber('end_complete', Complete, self.complete_callback)

    def pose_callback(self, data):
        self.pose_data = np.array(data.pose)

    def complete_callback(self, data):
        # 전체 미션이 종료되면 True로 설정
        if data.complete:
            self.if_end = True
        else:
            self.if_end = False


# If collide, Move a car to Initialpose
def collision_detection(lidar_data):
    min_distance = np.min(np.array(lidar_data) - np.array(check_array))
    if min_distance <= 0.01:
        return True
    else:
        return False

if __name__ == "__main__":
    rospy.init_node("Check_collision")
    rospy.Subscriber('/timer_start', Bool, timer_starter_callback)
    pose_pub = rospy.Publisher('pose', Pose, queue_size=1)
    control_pub = rospy.Publisher('drive', AckermannDrive, queue_size=1)
    respone_pub = rospy.Publisher('respone', Bool, queue_size=1)
    check_col = CheckCollide()
    check_end = CheckEnd()
    rate = rospy.Rate(param.thread_rate)

    # Get mission number
    map_number = rospy.get_param('~map_number')

    # Spawn list
    if map_number == 1:
        spawn_list = param.MAP_1_SPAWN_POINT
    elif map_number == 2:
        spawn_list = param.MAP_2_SPAWN_POINT
    elif map_number == 3:
        spawn_list = param.MAP_3_SPAWN_POINT
    elif map_number == 4:
        spawn_list = param.MAP_4_SPAWN_POINT
        rospy.loginfo("Jerk incoming!")
    elif map_number == 5:
        spawn_list = param.MAP_5_SPAWN_POINT
        rospy.loginfo("Jerk incoming!")
    else:
        rospy.loginfo("Incorrect map number.")

    time.sleep(1)

    if_terminated = False
    
    while not rospy.is_shutdown():
        if not if_terminated:
            # Check if CAR reaches the end point
            if check_end.if_end:
                # if END, show message
                rospy.loginfo("Time : %.3f", elapsed_time)
                rospy.loginfo("Collision : %d", check_col.collision_count)

                # Kill main node
                os.system("rosnode kill mission || true")
                if_terminated = True

                # Stop a car
                control_msg = AckermannDrive()
                control_msg.speed = 0
                control_pub.publish(control_msg)

            else:
                # else, check collision
                if collision_detection(check_col.lidar_data):
                    pose = Pose()
                    pose.position.x = spawn_list[check_col.spawn_index][0]
                    pose.position.y = spawn_list[check_col.spawn_index][1]
                    pose.position.z = 0
                    angle_yaw = spawn_list[check_col.spawn_index][2]
                    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion_from_euler(0,0,pi*angle_yaw/180)
                    pose_pub.publish(pose)
                    respone = Bool()
                    respone.data = True
                    respone_pub.publish(respone)
                    check_col.collision_count += 1
                else:
                    respone = Bool()
                    respone.data = False
                    respone_pub.publish(respone)

                # Show messages
                if timer_started:
                    # rospy.loginfo(rospy.get_time()-timer_stamp)
                    elapsed_time = round((rospy.get_time() - timer_stamp), 3)
                    text_1 = "Time : " + str(elapsed_time) + " sec"
                else:
                    text_1 = "Time : Waiting for /timer_starter"
                text_2 = "Collision : " + str(check_col.collision_count)

                img = np.full(shape=(100,400,3),fill_value=0,dtype=np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(img, text_1, (0, 30), font, 1, (255, 255, 255), 2)
                cv2.putText(img, text_2, (0, 70), font, 1, (255, 255, 255), 2)
                cv2.imshow("image",img)
                cv2.moveWindow('image', 0, 0)
                cv2.waitKey(1)

                rate.sleep()
        else:
            # Show mission complete message
            complete_text = "Mission Complete!"
            complete_img = np.full(shape=(150,650,3),fill_value=0,dtype=np.uint8)
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(complete_img, complete_text, (15, 85), font, 2, (255, 255, 255), 5)
            cv2.imshow("complete_img",complete_img)
            cv2.moveWindow('complete_img', 450, 195)
            cv2.waitKey(1)

            rate.sleep()

    cv2.destroyAllWindows()

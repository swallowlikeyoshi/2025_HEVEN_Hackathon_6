#!/usr/bin/env python3

import rospy
import tf

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool
from nav_msgs.msg import Path
from racecar_simulator.msg import CenterPose, Traffic, ParkingInfoList
from math import *
from goal import *
from parameter_list import Param
from tf.transformations import euler_from_quaternion

param = Param()


class Database():
    def __init__(self, lidar=True):
        # init node
        rospy.init_node('sensor_node')

        # sensor subscriber
        if lidar: rospy.Subscriber("/scan", LaserScan, self.lidar_callback, queue_size=1)
        rospy.Subscriber("/car_center", CenterPose, self.pose_callback, queue_size=1)
        rospy.Subscriber("/traffic", Traffic, self.traffic_callback, queue_size=1)
        rospy.Subscriber("/parking_info_list", ParkingInfoList, self.parking_callback)
        rospy.Subscriber("/cur_mission", String, self.mission_name_callback)
        # 초록색 빨간색 subsriber
        rospy.Subscriber("/traffic_random", Traffic, self.traffic_light_callback)
        # GP Sub
        rospy.Subscriber("/global_path", Path, self.globalpath_callback)
        # Respone Check
        rospy.Subscriber("/respone", Bool, self.respone_callback)
        # Which mission?
        self.current_mission = 0

        # Data
        self.lidar_data = None
        self.pose_data = [0,0,0] # x, y, yaw

        # Traffic mission
        self.stop_line = None
        self.traffic_remaining_time = 0
        self.traffic_light = ""

        # Parking mission
        self.target_park = param.MAP_1_PARKING_AREA
        self.parking_list = []

        # gp
        self.global_path = []
        self.gp_sub = False
        self.respone = False

    def lidar_callback(self, data=LaserScan):
        self.lidar_data = data.ranges

    def pose_callback(self, data=CenterPose):
        self.pose_data = data.pose
        
    def traffic_callback(self, data=Traffic):
        self.stop_line = data.traffic
        self.traffic_remaining_time = data.second

    def parking_callback(self, data=ParkingInfoList):
        info_element = []
        for info in data.InfoList:
            info_element = [info.x, info.y, info.yaw, info.park_num]
        self.parking_list = info_element

    def mission_name_callback(self, data):
        if data.data == "parking":
            self.current_mission = PARKING_SPOT

        elif data.data == "stop_line":
            self.current_mission = STOP_LINE

        else:
            self.current_mission = 0

    def traffic_light_callback(self, data):
        self.traffic_light = data.traffic

    def globalpath_callback(self, data):
        if self.gp_sub:
            return
        for pose in data.poses:
            temp = [pose.pose.position.x, pose.pose.position.y]
            self.global_path.append(temp)
        self.gp_sub = True

    def respone_callback(self, data):
        self.respone = data.data

if __name__ == "__main__":
    try:
        test_db = Database(lidar=True)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                rate.sleep()
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
    except rospy.ROSInterruptException:
        rospy.loginfo("Error!!!")
#!/usr/bin/env python3

import rospy

from math import pi
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from racecar_simulator.msg import Traffic
from parameter_list import Param
from tf.transformations import quaternion_from_euler

param = Param()
curr_traffic = "GREEN"

def MakeGoalMarker(map_number):
    if map_number == 1:
        m = param.m
        m.pose.position.x = param.END_POINT_X_1
        m.pose.position.y = param.END_POINT_Y_1
        m.pose.position.z = 0
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        return m

    else:
        rospy.loginfo("Map number is incorrect.")

def MakeTrafficMarker(map_number):
    global curr_traffic
    if map_number == 1:
        traffic_stop = Marker()
        traffic_stop.header.frame_id = "map"
        traffic_stop.ns = "traffic_stop"
        traffic_stop.type = Marker.LINE_STRIP
        traffic_stop.action = Marker.ADD
        # 정지선 검정색으로 표시 
        traffic_stop.color.r, traffic_stop.color.g, traffic_stop.color.b = 0, 0, 0
        traffic_stop.color.a = 1
        traffic_stop.scale.x = 0.1
        traffic_stop.scale.y = 0.1
        traffic_stop.scale.z = 0

        # 차선 폭에 따라서 STOP_LINE_SIZE 튜닝 요망
        l_point = Point()
        l_point.x = param.MAP_1_STOP_LINE_X_1
        l_point.y = param.MAP_1_STOP_LINE_Y_1 + param.STOP_LINE_SIZE / 2
        l_point.z = 0 
        r_point = Point()
        r_point.x = param.MAP_1_STOP_LINE_X_1
        r_point.y = param.MAP_1_STOP_LINE_Y_1 - param.STOP_LINE_SIZE / 2
        r_point.z = 0

        traffic_stop.points.append(l_point)
        traffic_stop.points.append(r_point)

        stop_sign = Marker()
        stop_sign.header.frame_id = "map"
        stop_sign.ns = "stop_sign"
        stop_sign.type = Marker.CYLINDER
        stop_sign.action = Marker.ADD

        # 현재 신호등 색을 바탕으로 색 표시 
        # 초록 신호등
        if curr_traffic == "GREEN":
            stop_sign.color.r, stop_sign.color.g, stop_sign.color.b = 0, 1, 0
        else:
            stop_sign.color.r, stop_sign.color.g, stop_sign.color.b = 0, 0, 0

        stop_sign.color.a = 1
        stop_sign.scale.x = param.SIZE_OF_TROPHY
        stop_sign.scale.y = param.SIZE_OF_TROPHY
        stop_sign.scale.z = 0
        stop_sign.pose.position.x = param.MAP_1_STOP_LINE_X_1 + 0.5
        stop_sign.pose.position.y = param.MAP_1_STOP_LINE_Y_1 + 0.2
        stop_sign.pose.position.z = 0
        stop_sign.pose.orientation.x = 0.0
        stop_sign.pose.orientation.y = 0.0
        stop_sign.pose.orientation.z = 0.0
        stop_sign.pose.orientation.w = 1.0

        # traffic_stop_2 = Marker()
        # traffic_stop_2.header.frame_id = "map"
        # traffic_stop_2.ns = "traffic_stop_2"
        # traffic_stop_2.type = Marker.LINE_STRIP
        # traffic_stop_2.action = Marker.ADD
        # traffic_stop_2.color.r, traffic_stop_2.color.g, traffic_stop_2.color.b = 1, 0, 0
        # traffic_stop_2.color.a = 1
        # traffic_stop_2.scale.x = 0.1
        # traffic_stop_2.scale.y = 0.1
        # traffic_stop_2.scale.z = 0
        # l_point = Point()
        # l_point.x = param.MAP_1_STOP_LINE_X_2 - param.STOP_LINE_SIZE / 2
        # l_point.y = param.MAP_1_STOP_LINE_Y_2
        # l_point.z = 0
        # r_point = Point()
        # r_point.x = param.MAP_1_STOP_LINE_X_2 + param.STOP_LINE_SIZE / 2
        # r_point.y = param.MAP_1_STOP_LINE_Y_2
        # r_point.z = 0

        # traffic_stop_2.points.append(l_point)
        # traffic_stop_2.points.append(r_point)

        stop_sign_2 = Marker()
        stop_sign_2.header.frame_id = "map"
        stop_sign_2.ns = "stop_sign_2"
        stop_sign_2.type = Marker.CYLINDER
        stop_sign_2.action = Marker.ADD
        # 빨간 신호등
        if curr_traffic == "RED":
            stop_sign_2.color.r, stop_sign_2.color.g, stop_sign_2.color.b = 1, 0, 0
        else:
            stop_sign_2.color.r, stop_sign_2.color.g, stop_sign_2.color.b = 0, 0, 0

        stop_sign_2.color.a = 1
        stop_sign_2.scale.x = param.SIZE_OF_TROPHY
        stop_sign_2.scale.y = param.SIZE_OF_TROPHY
        stop_sign_2.scale.z = 0
        stop_sign_2.pose.position.x = param.MAP_1_STOP_LINE_X_1 + 0.5
        stop_sign_2.pose.position.y = param.MAP_1_STOP_LINE_Y_1 - 0.2
        stop_sign_2.pose.position.z = 0
        stop_sign_2.pose.orientation.x = 0.0
        stop_sign_2.pose.orientation.y = 0.0
        stop_sign_2.pose.orientation.z = 0.0
        stop_sign_2.pose.orientation.w = 1.0
        
        return traffic_stop, stop_sign, stop_sign_2
    
    else:
        rospy.loginfo("Map number is incorrect.")

def ParkinglotMarker():
    m = Marker()
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.color.r, m.color.g, m.color.b = 0, 0, 1
    m.color.a = 1
    m.scale.x = param.PARKING_LOT_WIDTH
    m.scale.y = param.PARKING_LOT_HEIGHT
    m.scale.z = 0.1
    m.pose.position.z = 0
    m.pose.orientation.x = 0.0
    m.pose.orientation.y = 0.0
    m.pose.orientation.z = 0.0
    m.pose.orientation.w = 1.0

    return m
    
def MakeParkinglotMarker(map_number):
    if map_number == 1:
        tilt_degree_1 = param.PARKING_LOT_TILT_DEGREE_1 * pi / 180
        tilt_degree_2 = param.PARKING_LOT_TILT_DEGREE_2 * pi / 180
        tilt_degree_3 = param.PARKING_LOT_TILT_DEGREE_3 * pi / 180
        
        
        qu_x1, qu_y1, qu_z1, qu_w1 = quaternion_from_euler(0,0,tilt_degree_1)
        qu_x2, qu_y2, qu_z2, qu_w2 = quaternion_from_euler(0,0,tilt_degree_2)
        qu_x3, qu_y3, qu_z3, qu_w3 = quaternion_from_euler(0,0,tilt_degree_3)

        parking_lot_1 = ParkinglotMarker()
        parking_lot_1.header.frame_id = "map"
        parking_lot_1.ns = "lot_1"
        parking_lot_1.pose.position.x = param.MAP_1_PARKING_LOT_X_1
        parking_lot_1.pose.position.y = param.MAP_1_PARKING_LOT_Y_1
        parking_lot_1.pose.orientation.x = qu_x1
        parking_lot_1.pose.orientation.y = qu_y1
        parking_lot_1.pose.orientation.z = qu_z1
        parking_lot_1.pose.orientation.w = qu_w1

        parking_lot_2 = ParkinglotMarker()
        parking_lot_2.header.frame_id = "map"
        parking_lot_2.ns = "lot_2"
        parking_lot_2.pose.position.x = param.MAP_1_PARKING_LOT_X_2
        parking_lot_2.pose.position.y = param.MAP_1_PARKING_LOT_Y_2
        parking_lot_2.pose.orientation.x = qu_x2
        parking_lot_2.pose.orientation.y = qu_y2
        parking_lot_2.pose.orientation.z = qu_z2
        parking_lot_2.pose.orientation.w = qu_w2
        
        parking_lot_3 = ParkinglotMarker()
        parking_lot_3.header.frame_id = "map"
        parking_lot_3.ns = "lot_3"
        parking_lot_3.pose.position.x = param.MAP_1_PARKING_LOT_X_3
        parking_lot_3.pose.position.y = param.MAP_1_PARKING_LOT_Y_3
        parking_lot_3.pose.orientation.x = qu_x3
        parking_lot_3.pose.orientation.y = qu_y3
        parking_lot_3.pose.orientation.z = qu_z3
        parking_lot_3.pose.orientation.w = qu_w3

        return parking_lot_1, parking_lot_2, parking_lot_3

    else:
        return None

def traffic_callback(data) :
    global curr_traffic
    curr_traffic = data.traffic

if __name__ == "__main__":
    rospy.init_node("Visualize_node")
    visual_pub = rospy.Publisher('marker_array', MarkerArray, queue_size=1)
    traffic_sub = rospy.Subscriber('/traffic_random', Traffic, traffic_callback)
    rate = rospy.Rate(10)
    # Get mission number
    map_number = rospy.get_param('~map_number')

    goal_marker = MakeGoalMarker(map_number)
    parking_marker = MakeParkinglotMarker(map_number)

    while not rospy.is_shutdown():
        traffic_marker = MakeTrafficMarker(map_number)

        mkarray_msg = MarkerArray()
        temp_list = []
        temp_list.append(goal_marker)

        if traffic_marker is not None:
            for i in traffic_marker:
                temp_list.append(i)

        if parking_marker is not None:
            for i in parking_marker:
                temp_list.append(i)

        if len(temp_list) != 0:
            mkarray_msg = temp_list

            visual_pub.publish(mkarray_msg)
        
        else:
            pass

        rate.sleep()

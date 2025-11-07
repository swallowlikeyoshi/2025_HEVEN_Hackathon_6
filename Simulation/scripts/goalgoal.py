#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped

def goal_callback(msg):
    # 메시지에서 좌표 추출 및 출력
    x = msg.pose.position.x
    y = msg.pose.position.y
    rospy.loginfo(f"Received goal: x={x}, y={y}")

def main():
    rospy.init_node('goal_listener', anonymous=True)
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, goal_callback)
    rospy.loginfo("Goal listener node started. Click '2D Nav Goal' in RViz.")
    rospy.spin()

if __name__ == '__main__':
    main()

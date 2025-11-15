#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose

class InitialPoseRelay:
    def __init__(self):
        rospy.init_node('initialpose_to_pose_node')

        # Publisher
        self.pose_pub = rospy.Publisher('/pose', Pose, queue_size=10)

        # Subscriber
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.callback)

        rospy.loginfo("Node started: Subscribing to /initialpose and publishing /pose")
        rospy.spin()

    def callback(self, msg):
        pose_msg = Pose()
        pose_msg.position = msg.pose.pose.position
        pose_msg.orientation = msg.pose.pose.orientation

        self.pose_pub.publish(pose_msg)
        rospy.loginfo("Published /pose from /initialpose")

if __name__ == '__main__':
    try:
        InitialPoseRelay()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import csv
import os
import rospkg

class CSVToPath:
    def __init__(self):
        rospy.init_node('path_maker')
        
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('racecar_simulator')

        csv_file_path = os.path.join(package_path, 'csv', 'global_path.csv')

        self.path_pub = rospy.Publisher('/global_path', Path, queue_size=10)

        self.path = Path()
        self.path.header.frame_id = "map"

        self.load_path_from_csv(csv_file_path)

        rospy.Timer(rospy.Duration(1.0), self.publish_path)

    def load_path_from_csv(self, csv_file):
        if not os.path.exists(csv_file):
            rospy.logwarn(f"No CSV file found at {csv_file}")
            return

        with open(csv_file, mode='r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.pose.position.x = float(row[0])
                pose.pose.position.y = float(row[1])
                pose.pose.position.z = 0.0
                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = 0.0
                pose.pose.orientation.w = 1.0

                self.path.poses.append(pose)

        rospy.loginfo(f"Path loaded from {csv_file}")

    def publish_path(self, event):
        self.path.header.stamp = rospy.Time.now()
        self.path_pub.publish(self.path)

if __name__ == '__main__':
    try:
        CSVToPath()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

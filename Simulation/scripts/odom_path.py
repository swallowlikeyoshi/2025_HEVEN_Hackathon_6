#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
import csv
import os
import rospkg

class OdomToCSV:
    def __init__(self):
        rospy.init_node('odom_to_csv')
        
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('racecar_simulator')

        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.csv_file = os.path.join(package_path, 'csvs', 'odom_path.csv')
        self.csv_writer = None
        self.open_csv()

    def open_csv(self):
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        self.csv_file_handle = open(self.csv_file, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file_handle)
        self.csv_writer.writerow(['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        self.csv_writer.writerow([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])

    def close_csv(self):
        if self.csv_file_handle:
            self.csv_file_handle.close()
            rospy.loginfo(f"Path saved to {self.csv_file}")

    def __del__(self):
        self.close_csv()

if __name__ == '__main__':
    try:
        node = OdomToCSV()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

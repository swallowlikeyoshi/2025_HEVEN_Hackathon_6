#!/usr/bin/env python3
import rospy
import numpy as np
from fs_msgs.msg import Track, Cone
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
import tf
import os

class DNFNode:
    def __init__(self):
        # Cone storage
        self.blue_cones = []
        self.yellow_cones = []
        self.mid_points = []
        
        # Odometry [x, y, theta(deg), speed(m/s)]
        self.state = [0.0, 0.0, 0.0, 0.0]

         # DNF flag
        self.dnf_triggered = False
        
        # tf
        self.tf_broadcaster = tf.TransformBroadcaster()

        # Subscribers
        rospy.Subscriber("/fsds/testing_only/track", Track, self.track_callback)
        rospy.Subscriber("/fsds/testing_only/odom", Odometry, self.odom_callback)

    def track_callback(self, msg):

        # Separate cones
        self.blue_cones = []
        self.yellow_cones = []
        for cone in msg.track:
            if cone.color == Cone.BLUE:
                self.blue_cones.append(cone)
            elif cone.color == Cone.YELLOW:
                self.yellow_cones.append(cone)

        # Compute midpoints
        self.calculate_midpoints()

        rospy.loginfo(f"Processed path with {len(self.mid_points)} points.")

    def calculate_midpoints(self):
        self.mid_points = []
        num_pairs = min(len(self.blue_cones), len(self.yellow_cones))

        for i in range(num_pairs):
            bc = self.blue_cones[i]
            yc = self.yellow_cones[i]

            mid = Point()
            mid.x = (bc.location.x + yc.location.x) / 2.0
            mid.y = (bc.location.y + yc.location.y) / 2.0
            mid.z = (bc.location.z + yc.location.z) / 2.0
            self.mid_points.append(mid)

    def odom_callback(self, msg):

        # start = rospy.get_time()

        if self.dnf_triggered:
            return 
        # Extract position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        # Extract orientation (quaternion)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        _, _, yaw = tf.transformations.euler_from_quaternion([qx, qy, qz, qw])

        self.state[0] = x
        self.state[1] = y
        self.state[2] = np.degrees(yaw)

        dnf = self.check_dnf(x,y)

        # rospy.loginfo(rospy.get_time()-start)

        if dnf:
            rospy.logwarn("---------DNF---------")
            os.system("rosnode kill control || true")
            self.dnf_triggered = True 

    def check_dnf(self, car_x, car_y):
        """
        Returns True if the car is farther than 5 meters from the track path.
        The track is approximated as line segments connecting midpoints.
        """
        if len(self.mid_points) < 2:
            return False  # Not enough points to form a path

        def point_to_segment_distance(px, py, x1, y1, x2, y2):
            """Compute shortest distance from point (px, py) to line segment (x1, y1)-(x2, y2)"""
            # Vector from point 1 to point 2
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                # Segment is a point
                return np.hypot(px - x1, py - y1)

            # Project point onto line, computing parameter t
            t = ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)
            t = max(0, min(1, t))  # Clamp t to [0, 1] to stay within segment
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            return np.hypot(px - closest_x, py - closest_y)

        # Compute distance to all segments
        min_distance = float('inf')
        for i in range(len(self.mid_points)):
            p1 = self.mid_points[i]
            if i == len(self.mid_points)-1:
                p2 = self.mid_points[0]
            else:
                p2 = self.mid_points[i + 1]
            dist = point_to_segment_distance(car_x, car_y, p1.x, p1.y, p2.x, p2.y)
            if dist < min_distance:
                min_distance = dist

        return min_distance > 5.0


def main():
    rospy.init_node("dnf_node")
    node = DNFNode()
    rospy.spin()

if __name__ == "__main__":
    main()

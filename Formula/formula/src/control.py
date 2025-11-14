#!/usr/bin/env python3
import rospy
import numpy as np
from fs_msgs.msg import Track, Cone, ControlCommand
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point, TwistWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import tf

def cone_color(cone_type):
    if cone_type == Cone.BLUE:
        return ColorRGBA(0.0, 0.0, 1.0, 1.0)
    elif cone_type == Cone.YELLOW:
        return ColorRGBA(1.0, 1.0, 0.0, 1.0)
    elif cone_type == Cone.ORANGE_BIG or cone_type == Cone.ORANGE_SMALL:
        return ColorRGBA(1.0, 0.5, 0.0, 1.0)
    else:
        return ColorRGBA(1.0, 1.0, 1.0, 1.0)  # fallback

class ControlNode:
    def __init__(self):
        # Cone storage
        self.blue_cones = []
        self.yellow_cones = []
        self.mid_points = []
        
        # Cone visualization
        self.cone_markers = MarkerArray()
        
        # Odometry [x, y, theta(deg), speed(m/s)]
        self.state = [0.0, 0.0, 0.0, 0.0]
        
        # tf
        self.tf_broadcaster = tf.TransformBroadcaster()

        # Subscribers
        rospy.Subscriber("/fsds/testing_only/track", Track, self.track_callback)
        rospy.Subscriber("/fsds/testing_only/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/fsds/gss", TwistWithCovarianceStamped, self.speed_callback)

        # Publishers
        self.cmd_pub = rospy.Publisher("/fsds/control_command", ControlCommand, queue_size=10)
        self.midline_path_pub = rospy.Publisher("/midpoint_path", Path, queue_size=10)
        self.cones_pub = rospy.Publisher("/cones_markers", MarkerArray, queue_size=10)

        

    def track_callback(self, msg):

        self.cone_markers = self.track_to_markers(msg)

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
    
    def track_to_markers(self, track_msg, frame_id="fsds/map"):
        markers = MarkerArray()
        for i, cone in enumerate(track_msg.track):
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = rospy.Time(0)
            m.ns = "cones"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = cone.location
            m.pose.orientation.w = 1.0
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.3
            m.color = cone_color(cone.color)
            m.lifetime = rospy.Duration(0.5)
            markers.markers.append(m)
        return markers

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

    def publish_midpoints(self):
        path_msg = Path()
        path_msg.header.frame_id = "fsds/map" 
        path_msg.header.stamp = rospy.Time.now()

        for pt in self.mid_points:
            pose = PoseStamped()
            pose.header.frame_id = path_msg.header.frame_id
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position = pt
            pose.pose.orientation.w = 1.0 
            path_msg.poses.append(pose)

        self.midline_path_pub.publish(path_msg)

    def odom_callback(self, msg):
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

        # Publish TF from "fsds/map" to "fsds/FSCar"
        self.tf_broadcaster.sendTransform(
            (x, y, z),
            (qx, qy, qz, qw),
            rospy.Time.now(),
            "fsds/FSCar",  # child frame
            "fsds/map"     # parent frame
        )

    def speed_callback(self, msg):
       xv = msg.twist.twist.linear.x
       yv = msg.twist.twist.linear.y

       self.state[3] = np.hypot(xv, yv)

    def run(self):
        x, y, theta, v = self.state     # car state: x(m), y(m), theta(deg), v(m/s)
        path = self.mid_points          # path[i].x = x-coordinate of point i
                                        # path[i].y = y-coordinate of point i

        ############ Write your control algorithm here ############

        



        throttle = 0.0
        steering = 0.0
        brake = 0.0

        ###########################################################

        cmd = ControlCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.throttle = throttle
        cmd.steering = steering
        cmd.brake = brake
        self.cmd_pub.publish(cmd)

        # Publish midpoints as Path
        self.publish_midpoints()

        # Publish cones as MarkerArray
        self.cones_pub.publish(self.cone_markers)

def main():
    rospy.init_node("control_node")
    node = ControlNode()

    rate = rospy.Rate(30)  # 30 Hz
    while not rospy.is_shutdown():
        node.run()
        rate.sleep()

if __name__ == "__main__":
    main()

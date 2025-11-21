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

        

    # def track_callback(self, msg):

    #     self.cone_markers = self.track_to_markers(msg)

    #     # Separate cones
    #     self.blue_cones = []
    #     self.yellow_cones = []
    #     for cone in msg.track:
    #         if cone.color == Cone.BLUE:
    #             self.blue_cones.append(cone)
    #         elif cone.color == Cone.YELLOW:
    #             self.yellow_cones.append(cone)

    #     # Compute midpoints
    #     self.calculate_midpoints()

    #     rospy.loginfo(f"Processed path with {len(self.mid_points)} points.")
    
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

        



        throttle = 0.0  # 악셀 (0.0~1.0)
        steering = 0.0  # 조향각 (-1.0~1.0)
        brake = 0.0     # 브레이크 (0.0~1.0)

        throttle, steering, brake = self.pure_pursuit(lookahead=3.0, wheelbase=1.5, target_speed=5.0)

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

    # ==========================================================
    # [NEW] Step 1: 거리 기반 경로 생성 함수 추가
    # ==========================================================
    def calculate_midpoints_robust(self):
        """
        Blue 콘을 기준으로 가장 가까운 Yellow 콘을 찾아 짝을 짓는 방식.
        인덱스 밀림 현상으로 인한 지그재그 경로를 방지함.
        """
        self.mid_points = []
        
        if not self.blue_cones or not self.yellow_cones:
            return

        # 파란 콘을 하나씩 순회하며 짝을 찾음
        for bc in self.blue_cones:
            min_dist = float('inf')
            closest_yc = None

            # 모든 노란 콘 중에서 가장 가까운 것 탐색
            for yc in self.yellow_cones:
                dist = np.hypot(bc.location.x - yc.location.x, bc.location.y - yc.location.y)
                if dist < min_dist:
                    min_dist = dist
                    closest_yc = yc
            
            # [중요] 트랙 폭(약 5~8m)보다 너무 멀리 있는 콘은 짝이 아님 (오류 방지)
            # 해커톤 트랙 규격을 고려해 10m 이내일 때만 유효한 게이트로 인정
            if closest_yc is not None and min_dist < 10.0:
                mid = Point()
                mid.x = (bc.location.x + closest_yc.location.x) / 2.0
                mid.y = (bc.location.y + closest_yc.location.y) / 2.0
                mid.z = (bc.location.z + closest_yc.location.z) / 2.0
                self.mid_points.append(mid)

        # (선택 사항) 차량과 가까운 순서대로 경로 정렬 (주행 순서 보장)
        # 이 부분이 없으면 경로 점들의 순서가 뒤죽박죽일 수 있음
        x, y, _, _ = self.state
        self.mid_points.sort(key=lambda p: np.hypot(p.x - x, p.y - y))


    # ==========================================================
    # [MODIFY] track_callback 수정
    # ==========================================================
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

        # [변경] 기존 calculate_midpoints() 대신 새로운 로직 사용
        # self.calculate_midpoints()  <-- 기존 함수 주석 처리
        self.calculate_midpoints_robust() # <-- 새로운 함수 호출

        rospy.loginfo(f"Processed path with {len(self.mid_points)} points.")


    # ==========================================================
    # [MODIFY] Pure Pursuit 제어 로직 고도화 (이전 답변 내용)
    # ==========================================================
    def pure_pursuit(self, lookahead=None, wheelbase=1.5, target_speed=None):
        x, y, theta_deg, v = self.state
        
        if not self.mid_points:
            return 0.0, 0.0, 1.0 

        # 1. Adaptive Lookahead (속도에 비례해 멀리 봄)
        k_gain = 0.15
        min_lookahead = 2.5
        adaptive_lookahead = min_lookahead + (k_gain * v)

        # 2. 경로상에서 가장 적절한 목표점 찾기
        # 현재 차량 위치에서 각 점까지의 거리를 계산
        dists = [np.hypot(pt.x - x, pt.y - y) for pt in self.mid_points]
        min_dist_idx = np.argmin(dists)
        
        target_point = self.mid_points[min_dist_idx]
        
        # 가장 가까운 점부터 시작해서 lookahead 거리보다 먼 첫 번째 점을 선택
        for i in range(min_dist_idx, len(self.mid_points)):
            if dists[i] > adaptive_lookahead:
                target_point = self.mid_points[i]
                break
        
        # 3. Steering 계산
        target_dx = target_point.x - x
        target_dy = target_point.y - y
        target_theta = np.arctan2(target_dy, target_dx)
        theta_rad = np.radians(theta_deg)
        alpha = (target_theta - theta_rad + np.pi) % (2 * np.pi) - np.pi

        steering = np.arctan2(2 * wheelbase * np.sin(alpha), adaptive_lookahead)
        steering = np.clip(steering, -1.0, 1.0)

        # 4. Speed Profile (조향각 기반 감속)
        MAX_SPEED = 10.0
        MIN_SPEED = 4.0
        
        # 핸들을 많이 꺾을수록(커브) 속도를 줄임
        target_v = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * (abs(steering))
        target_v = max(MIN_SPEED, target_v)

        # 5. PID 제어 (간략화)
        throttle = 0.0
        brake = 0.0
        speed_error = target_v - v

        if speed_error > 0:
            throttle = np.clip(0.5 * speed_error, 0.0, 1.0)
        else:
            # 속도가 너무 빠르면 브레이크
            # 단, 미세한 차이는 무시(히스테리시스)하여 진동 방지
            if speed_error < -0.5:
                brake = np.clip(0.2 * abs(speed_error), 0.0, 0.5)
                throttle = 0.0

        return throttle, steering, brake

def main():
    rospy.init_node("control_node")
    node = ControlNode()

    rate = rospy.Rate(30)  # 30 Hz
    while not rospy.is_shutdown():
        node.run()
        rate.sleep()

if __name__ == "__main__":
    main()

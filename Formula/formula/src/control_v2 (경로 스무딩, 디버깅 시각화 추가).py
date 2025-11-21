#!/usr/bin/env python3
import rospy
import numpy as np
from fs_msgs.msg import Track, Cone, ControlCommand
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point, TwistWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import tf

# 추가된 라이브러리

from scipy.interpolate import splprep, splev 
from fs_msgs.msg import Track, Cone, ControlCommand
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point, TwistWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import tf

# ==============================================================================
# [TUNING] 하이퍼파라미터 (여기서 값을 바꾸면 전체 로직에 적용됩니다)
# ==============================================================================

# 1. 경로 생성 관련
TRACK_MATCH_DIST = 10.0    # 파란 콘-노란 콘 매칭 최대 거리 (m)
PATH_SMOOTHING_FACTOR = 0.5 # 경로 스무딩 강도 (0: 원래 점 유지, 클수록 부드러움)
PATH_DENSITY = 5           # 스무딩 시 점을 몇 배로 늘릴지 (조밀할수록 부드러움)

# 2. Pure Pursuit (조향) 관련
WHEELBASE = 1.5            # 차량 축거 (m)
LOOKAHEAD_MIN = 2.5        # 최소 전방 주시 거리 (m)
LOOKAHEAD_GAIN = 0.15      # 속도 비례 주시 거리 증가율 (Lookahead = MIN + GAIN * V)

# 3. 속도 제어 (Speed Profile) 관련
MAX_SPEED = 12.0           # 직선 구간 최대 속도 (m/s)
MIN_SPEED = 4.0            # 코너 구간 최소 속도 (m/s)
CORNER_STIFFNESS = 1.0     # 코너 감속 민감도 (클수록 조금만 꺾어도 감속)

# 4. PID 제어 게인
K_ACCEL = 0.6              # 가속 P 게인
K_BRAKE = 0.4              # 브레이크 P 게인

# ==============================================================================

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

        # [NEW] 목표 지점(Target Point) 시각화용 Publisher 추가
        self.target_pub = rospy.Publisher("/debug/target_point", Marker, queue_size=10)
    
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

        # 🎯 Rviz 설정 방법 (중요!)
        # 코드를 실행한 후 Rviz에서 다음을 추가해야 눈에 보입니다.
        # Rviz 실행
        # 왼쪽 하단 [Add] 버튼 클릭
        # Marker 선택 -> Topic을 /debug/target_point로 설정 (초록색 공)
        # Path 선택 -> Topic을 /midpoint_path로 설정 (경로 선)
        # Tip: Path의 Color를 빨간색(255, 0, 0)으로 바꾸면 잘 보입니다.

        # 👀 디버깅 포인트
        # 시뮬레이션을 돌리면서 초록색 공을 유심히 보세요.
        # 직선: 초록색 공이 차보다 훨씬 앞(약 5~10m)에 있어야 합니다. (속도가 빠르니까)
        # 코너: 초록색 공이 차 쪽으로 가까워져야 합니다. (속도가 줄고 더 정교하게 돌아야 하니까)
        # 만약 초록색 공이 트랙 밖으로 튄다면? -> calculate_midpoints_robust의 매칭 거리가 너무 길거나, 노이즈가 낀 것입니다.

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

    def publish_target_marker(self, target_point):
        """
        Pure Pursuit이 바라보는 목표 지점에 '초록색 공'을 띄움
        """
        marker = Marker()
        marker.header.frame_id = "fsds/map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "target_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # 위치 설정
        marker.pose.position = target_point
        marker.pose.orientation.w = 1.0
        
        # 크기 (눈에 잘 띄게 0.5m 크기로 설정)
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        
        # 색상 (밝은 초록색)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0 # 투명도 (1.0 = 불투명)

        self.target_pub.publish(marker)

    # [Step 1 & 2] 경로 생성 및 스무딩 통합
    def track_callback(self, msg):
        self.cone_markers = self.track_to_markers(msg)

        self.blue_cones = []
        self.yellow_cones = []
        for cone in msg.track:
            if cone.color == Cone.BLUE:
                self.blue_cones.append(cone)
            elif cone.color == Cone.YELLOW:
                self.yellow_cones.append(cone)

        # 1. 견고한 중간점 계산 (Step 1)
        self.calculate_midpoints_robust()
        
        # 2. 경로 스무딩 적용 (Step 2 - NEW!)
        self.smooth_path()

        # rospy.loginfo(f"Path points: {len(self.mid_points)}")

    def calculate_midpoints_robust(self):
        """거리 기반 매칭 로직 (상수 적용)"""
        self.mid_points = []
        if not self.blue_cones or not self.yellow_cones:
            return

        for bc in self.blue_cones:
            min_dist = float('inf')
            closest_yc = None

            for yc in self.yellow_cones:
                dist = np.hypot(bc.location.x - yc.location.x, bc.location.y - yc.location.y)
                if dist < min_dist:
                    min_dist = dist
                    closest_yc = yc
            
            # 상단 상수(TRACK_MATCH_DIST) 사용
            if closest_yc is not None and min_dist < TRACK_MATCH_DIST:
                mid = Point()
                mid.x = (bc.location.x + closest_yc.location.x) / 2.0
                mid.y = (bc.location.y + closest_yc.location.y) / 2.0
                mid.z = (bc.location.z + closest_yc.location.z) / 2.0
                self.mid_points.append(mid)

        # 거리순 정렬
        x, y, _, _ = self.state
        self.mid_points.sort(key=lambda p: np.hypot(p.x - x, p.y - y))

    def smooth_path(self):
        """
        [Step 2] B-Spline을 이용한 경로 스무딩
        거친 점들을 부드러운 곡선으로 변환하고 점의 개수를 늘림
        """
        # 점이 너무 적으면 스무딩 불가 (최소 4개 필요 for cubic spline)
        if len(self.mid_points) < 4:
            return

        try:
            # 1. x, y 좌표 추출
            x_pts = [p.x for p in self.mid_points]
            y_pts = [p.y for p in self.mid_points]

            # 중복 점 제거 (Spline 계산 시 에러 방지)
            # 아주 가까운 점들이 겹쳐 있으면 보간법이 실패할 수 있음
            okay_indices = [0]
            for i in range(1, len(x_pts)):
                if np.hypot(x_pts[i]-x_pts[i-1], y_pts[i]-y_pts[i-1]) > 0.1:
                    okay_indices.append(i)
            
            if len(okay_indices) < 4:
                return
                
            x_pts = [x_pts[i] for i in okay_indices]
            y_pts = [y_pts[i] for i in okay_indices]

            # 2. B-Spline 표현식 계산 (tck)
            # s: smoothing factor (상수 사용)
            tck, u = splprep([x_pts, y_pts], s=PATH_SMOOTHING_FACTOR)

            # 3. 더 조밀한 점 생성 (Interpolation)
            # 기존 점 개수보다 PATH_DENSITY배 만큼 더 많이 생성
            u_new = np.linspace(0, 1, num=len(x_pts) * PATH_DENSITY)
            x_new, y_new = splev(u_new, tck)

            # 4. self.mid_points 업데이트
            new_points = []
            for i in range(len(x_new)):
                p = Point()
                p.x = x_new[i]
                p.y = y_new[i]
                p.z = 0.0 # z는 평지 가정
                new_points.append(p)
            
            self.mid_points = new_points

        except Exception as e:
            rospy.logwarn(f"Spline smoothing failed: {e}")

    def pure_pursuit(self, lookahead=None, wheelbase=None, target_speed=None):
        """
        상수(CONSTANTS)를 적용한 Pure Pursuit
        """
        x, y, theta_deg, v = self.state
        
        if not self.mid_points:
            return 0.0, 0.0, 1.0 

        # [상수 적용] Adaptive Lookahead
        adaptive_lookahead = LOOKAHEAD_MIN + (LOOKAHEAD_GAIN * v)

        # 최적 목표점 탐색
        dists = [np.hypot(pt.x - x, pt.y - y) for pt in self.mid_points]
        min_dist_idx = np.argmin(dists)
        
        target_point = self.mid_points[min_dist_idx]
        for i in range(min_dist_idx, len(self.mid_points)):
            if dists[i] > adaptive_lookahead:
                target_point = self.mid_points[i]
                break
        
        # 조향각 계산
        target_dx = target_point.x - x
        target_dy = target_point.y - y
        target_theta = np.arctan2(target_dy, target_dx)
        theta_rad = np.radians(theta_deg)
        alpha = (target_theta - theta_rad + np.pi) % (2 * np.pi) - np.pi

        # [상수 적용] Wheelbase
        steering = np.arctan2(2 * WHEELBASE * np.sin(alpha), adaptive_lookahead)
        steering = np.clip(steering, -1.0, 1.0)

        # [상수 적용] Dynamic Speed Profile
        # abs(steering)에 제곱 등을 적용해 민감도 조절 가능
        target_v = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * (abs(steering) ** CORNER_STIFFNESS)
        target_v = max(MIN_SPEED, target_v)

        # [상수 적용] PID Control
        throttle = 0.0
        brake = 0.0
        speed_error = target_v - v

        if speed_error > 0:
            throttle = np.clip(K_ACCEL * speed_error, 0.0, 1.0)
        else:
            if speed_error < -0.5:
                brake = np.clip(K_BRAKE * abs(speed_error), 0.0, 0.5)
                throttle = 0.0

        # [NEW] 디버깅: 목표 지점 시각화 호출
        self.publish_target_marker(target_point)

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

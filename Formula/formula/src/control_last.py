#!/usr/bin/env python3
import rospy
import numpy as np
from fs_msgs.msg import Track, Cone, ControlCommand
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point, TwistWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import tf

# ==============================================================================
#
# 1. 경로 생성
# 1.1 경로 생성 시 짝짓기 로직 개선
# 1.2 속도 기반 적응형 Lookahead 거리 적용
# 2. 경로 스무딩
# 2.1 B-Spline 보간법 적용
# 3. 경로 시각화
#
# 3. 속도 프로파일링: 직선에서는 빠르게, 코너에서는 느리게. 경로의 곡률 기반. 코너 진입 전에 미리 감속해야만 함!
# 4. Adaptive Lookahead: 속도가 빠를수록 더 먼 지점을 바라보도록 조정. 혹은 스탠리 메소드 활용.
# 5. 파일런 기억 및 활용: 이전 프레임의 파일런 정보를 저장하고, 현재 프레임의 파일런 감지에 활용.
#
# ==============================================================================

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
WHEELBASE = 1.55            # 차량 축거 (m)
LOOKAHEAD_MIN = 2.5        # 최소 전방 주시 거리 (m)
LOOKAHEAD_GAIN = 0.15      # 속도 비례 주시 거리 증가율 (Lookahead = MIN + GAIN * V)

# 3. 속도 제어 (Speed Profile) 관련

# 최고 속도: 27m/s (약 97km/h) (나중에 천천히 올리기!)
MAX_SPEED = 12.0           # 직선 구간 최대 속도 (m/s)
MIN_SPEED = 4.0            # 코너 구간 최소 속도 (m/s)
CORNER_STIFFNESS = 1.0     # 코너 감속 민감도 (클수록 조금만 꺾어도 감속)

# 4. PID 제어 게인
K_ACCEL = 0.6              # 가속 P 게인
K_BRAKE = 0.4              # 브레이크 P 게인

# [NEW Step 3] 예측형 속도 제어 관련
PREDICT_STEPS = 8          # 몇 개의 점 앞까지 미리 내다볼 것인가 (경로 탐색 범위)
CURVATURE_THRESHOLD = 0.15 # 이 값보다 곡률이 크면 급커브로 인식
BRAKE_LOOKAHEAD = 1.2      # 곡률 기반 감속 강도 (클수록 코너 앞에서 더 강하게 브레이크)

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
        x, y, theta_deg, v = self.state
        
        if not self.mid_points or len(self.mid_points) < 3:
            return 0.0, 0.0, 1.0 

        # 1. Adaptive Lookahead
        adaptive_lookahead = LOOKAHEAD_MIN + (LOOKAHEAD_GAIN * v)

        # 2. 목표점 탐색
        dists = [np.hypot(pt.x - x, pt.y - y) for pt in self.mid_points]
        min_dist_idx = np.argmin(dists)
        
        target_index = min_dist_idx
        for i in range(min_dist_idx, len(self.mid_points)):
            if dists[i] > adaptive_lookahead:
                target_index = i
                break
        
        target_point = self.mid_points[target_index]

        # 3. [NEW] 미래 경로 곡률 스캔 (Predictive Check)
        # 현재 목표점부터 미래의 몇 단계(PREDICT_STEPS) 앞까지 곡률을 미리 검사
        max_curvature = 0.0
        
        start_scan = target_index
        end_scan = min(len(self.mid_points) - 1, start_scan + PREDICT_STEPS)

        # 3개의 점씩 묶어서 곡률 계산
        for i in range(start_scan, end_scan - 2):
            p1 = self.mid_points[i]
            p2 = self.mid_points[i+1]
            p3 = self.mid_points[i+2]
            
            k = self.calculate_path_curvature(p1, p2, p3)
            if k > max_curvature:
                max_curvature = k

        # 4. 조향각(Steering) 계산
        target_dx = target_point.x - x
        target_dy = target_point.y - y
        target_theta = np.arctan2(target_dy, target_dx)
        theta_rad = np.radians(theta_deg)
        alpha = (target_theta - theta_rad + np.pi) % (2 * np.pi) - np.pi

        steering = np.arctan2(2 * WHEELBASE * np.sin(alpha), adaptive_lookahead)
        steering = np.clip(steering, -1.0, 1.0)

        # 5. [NEW] 곡률 기반 목표 속도 설정
        # 곡률이 클수록(급커브) 목표 속도를 낮춤. steering 값도 함께 고려.
        
        # 기본적으로 현재 조향각에 따라 감속
        curvature_speed = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * (abs(steering) ** CORNER_STIFFNESS)
        
        # 미래의 급커브가 감지되면 강력하게 미리 감속
        if max_curvature > CURVATURE_THRESHOLD:
            # 예측된 커브가 심할수록 속도를 더 많이 줄임
            predicted_speed = MAX_SPEED / (1.0 + BRAKE_LOOKAHEAD * max_curvature)
            # 현재 조향 기반 속도와 예측 속도 중 더 낮은(안전한) 속도 선택
            target_v = min(curvature_speed, predicted_speed, max(MIN_SPEED, predicted_speed))
        else:
            target_v = curvature_speed

        target_v = max(MIN_SPEED, target_v)

        # 6. PID Control (Throttle/Brake)
        throttle = 0.0
        brake = 0.0
        speed_error = target_v - v

        if speed_error > 0:
            throttle = np.clip(K_ACCEL * speed_error, 0.0, 1.0)
        else:
            # 감속이 필요할 때
            if speed_error < -0.5:
                brake = np.clip(K_BRAKE * abs(speed_error), 0.0, 0.8)
                throttle = 0.0

        # 디버깅용 시각화 (초록공)
        self.publish_target_marker(target_point)

        # (선택) 디버깅 로그: 현재 곡률과 목표 속도 확인
        # rospy.loginfo(f"Curvature: {max_curvature:.3f} | Target V: {target_v:.2f}")

        return throttle, steering, brake
        
    def calculate_path_curvature(self, p1, p2, p3):
        """
        세 점 (p1, p2, p3)을 지나는 외접원의 곡률(1/R)을 계산 (Menger Curvature)
        리턴값이 클수록 급커브
        """
        # 삼각형의 넓이 공식을 이용
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y

        # 세 변의 길이
        a = np.hypot(x1 - x2, y1 - y2)
        b = np.hypot(x2 - x3, y2 - y3)
        c = np.hypot(x3 - x1, y3 - y1)

        # 0으로 나누기 방지
        if (a * b * c) == 0:
            return 0.0

        # 헤론의 공식으로 삼각형 넓이(Area) 계산
        s = (a + b + c) / 2.0
        area = np.sqrt(abs(s * (s - a) * (s - b) * (s - c)))

        # 외접원 반경 R = (abc) / (4 * Area)
        # 곡률 k = 1 / R = (4 * Area) / (abc)
        curvature = (4 * area) / (a * b * c)
        return curvature

def main():
    rospy.init_node("control_node")
    node = ControlNode()

    rate = rospy.Rate(30)  # 30 Hz
    while not rospy.is_shutdown():
        node.run()
        rate.sleep()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import rospy
import numpy as np
from scipy.interpolate import splprep, splev 
from scipy.spatial.distance import cdist

from fs_msgs.msg import Track, Cone, ControlCommand
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point, TwistWithCovarianceStamped, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import tf

# 언더스티어가 나면 만져봐야 할 파라미터들
# MAX_BRAKE 1.0으로
# 

# ==============================================================================
# [TUNING] 하이퍼파라미터 설정 및 튜닝 가이드
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. 경로 생성 (Path Planning & Mapping)
# ------------------------------------------------------------------------------
# 좌우 콘을 매칭하여 중심선을 찾을 때 허용하는 최대 거리입니다.
# - 효과: 너무 작으면 트랙 폭이 넓은 곳에서 경로가 끊기고, 너무 크면 U턴 구간에서 반대편 콘을 잘못 연결할 수 있습니다.
# - 범위: 트랙 폭(약 4~5m)보다 약간 크게 설정 (6.0 ~ 12.0)
TRACK_MATCH_DIST = 8.0

# Spline 보간 시 경로의 부드러운 정도를 결정합니다. (scipy splprep의 's' 파라미터)
# - 효과: 0에 가까우면 콘 위치를 그대로 지나가려 해서 경로가 구불거립니다. 크면 경로는 부드러워지지만 트랙 중심에서 벗어날 수 있습니다.
# - 범위: 0.0 (원본 유지) ~ 1.0 (매우 부드러움)
PATH_SMOOTHING_FACTOR = 0.5

# 생성된 경로 점들의 밀도입니다. (콘과 콘 사이에 몇 개의 점을 찍을지)
# - 효과: 높을수록 정밀한 제어가 가능하지만 연산량이 늘어납니다. Pure Pursuit은 점이 촘촘해야 부드럽게 작동합니다.
# - 범위: 3 ~ 10
PATH_DENSITY = 8

# 같은 색상의 콘을 연결할 때 허용하는 최대 거리입니다.
# - 효과: 콘이 하나 누락되었을 때 다음 콘을 연결할지, 아니면 여기서 끊을지 결정합니다.
# - 범위: 5.0 ~ 15.0
MAX_CONE_CONNECT_DIST = 10.0


# ------------------------------------------------------------------------------
# 2. 차량 제원 (Vehicle Specs)
# ------------------------------------------------------------------------------
# 앞바퀴 축과 뒷바퀴 축 사이의 거리(m)입니다.
# - 중요: 이 값이 실제 차량과 다르면 조향각 계산(Atan)이 틀어져서 코너를 제대로 못 돕니다.
WHEELBASE = 1.55


# ------------------------------------------------------------------------------
# 3. 조향 제어 (Pure Pursuit Steering) - *튜닝 핵심*
# ------------------------------------------------------------------------------
# 속도가 0일 때의 기본 전방 주시 거리(Lookahead Distance)입니다.
# - 효과: 값이 작으면 핸들을 급하게 꺾어 진동(Oscillation)이 생기고, 값이 크면 코너를 안쪽으로 너무 파고들거나 반응이 느립니다.
# - 범위: 2.0 ~ 4.0
LOOKAHEAD_MIN = 2.0

# 속도에 비례하여 주시 거리를 늘리는 계수입니다. (Lookahead = Min + Gain * Speed)
# - 효과: 고속 주행 시 더 멀리 보고 미리 조향하게 하여 안정성을 높입니다.
# - 범위: 0.1 ~ 0.5
LOOKAHEAD_GAIN = 0.15


# ------------------------------------------------------------------------------
# 4. 속도 제어 (Longitudinal Control)
# ------------------------------------------------------------------------------
# 직선 구간에서 낼 수 있는 최대 목표 속도(m/s)입니다.
MAX_SPEED = 27.0

# 급코너 등 감속해야 할 상황에서의 최소 목표 속도(m/s)입니다.
MIN_SPEED = 1.0

# 코너 주행 시 속도를 줄이는 강도입니다. (물리적 타이어 마찰 한계 고려)
# - 효과: 높을수록 코너에서 더 많이 감속하여 안전하게 돕니다. (현재 코드 로직에 반영 필요)
CORNER_STIFFNESS = 2.0

# 가속 페달(Throttle) P제어 게인입니다.
# - 효과: 너무 크면 휠스핀이 발생하거나 속도가 요동치고, 너무 작으면 가속이 답답합니다.
# - 범위: 0.5 ~ 2.0
K_ACCEL = 0.8
MAX_ACCEL = 1.0

# 브레이크(Brake) P제어 게인입니다.
# - 효과: 너무 크면 바퀴가 잠길(Lock-up) 수 있습니다.
# - 범위: 0.5 ~ 1.5
K_BRAKE = 0.8
MAX_BRAKE = 1.0

# 조향각(Steering Angle)이 이 값보다 크면 '급커브'로 인식하여 감속합니다.
# - 효과: 0.0 ~ 1.0 사이의 값 (1.0이 최대 조향). 값을 낮추면 조금만 핸들을 꺾어도 속도를 줄입니다.
# - 범위: 0.2 ~ 0.5
STEERING_THRESHOLD = 0.25


# ------------------------------------------------------------------------------
# 5. 미래 경로 예측 및 감속 (Advanced)
# ------------------------------------------------------------------------------
# 현재 위치에서 몇 번째 앞의 경로점까지 곡률(Curvature)을 계산할지 결정합니다.
PREDICT_STEPS = 5

# 경로의 곡률이 이 값 이상이면 감속을 시작합니다.
# - 효과: 값이 작을수록 직선에 가까운 완만한 커브에서도 미리 감속합니다.
CURVATURE_THRESHOLD = 0.2

# 곡률 기반 감속 시, 차보다 얼마나 앞선 지점의 곡률을 볼 것인지(시간 기준) 설정합니다.
BRAKE_LOOKAHEAD = 1.2

# ------------------------------------------------------------------------------

def cone_color(cone_type):
    if cone_type == Cone.BLUE:
        return ColorRGBA(0.0, 0.0, 1.0, 1.0)
    elif cone_type == Cone.YELLOW:
        return ColorRGBA(1.0, 1.0, 0.0, 1.0)
    elif cone_type == Cone.ORANGE_BIG or cone_type == Cone.ORANGE_SMALL:
        return ColorRGBA(1.0, 0.5, 0.0, 1.0)
    else:
        return ColorRGBA(1.0, 1.0, 1.0, 1.0)

class ControlNode:
    def __init__(self):
        # Data storage
        self.blue_cones = []
        self.yellow_cones = []
        self.mid_points = []
        self.state = [0.0, 0.0, 0.0, 0.0] # x, y, theta, v

        # Visualization storage
        self.cone_markers = MarkerArray()

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

        # [DEBUG] Publishers for Visualization
        self.target_pub = rospy.Publisher("/debug/target_point", Marker, queue_size=10)
        self.blue_bound_pub = rospy.Publisher("/debug/blue_boundary", Marker, queue_size=10)
        self.yellow_bound_pub = rospy.Publisher("/debug/yellow_boundary", Marker, queue_size=10)
        self.text_info_pub = rospy.Publisher("/debug/car_info", Marker, queue_size=10)
        self.steering_arrow_pub = rospy.Publisher("/debug/steering_arrow", Marker, queue_size=10)

    # --------------------------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------------------------
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
            m.scale = Vector3(0.3, 0.3, 0.3)
            m.color = cone_color(cone.color)
            m.lifetime = rospy.Duration(0.5)
            markers.markers.append(m)
        return markers

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        _, _, yaw = tf.transformations.euler_from_quaternion([qx, qy, qz, qw])

        self.state[0] = x
        self.state[1] = y
        self.state[2] = np.degrees(yaw)

        self.tf_broadcaster.sendTransform(
            (x, y, z),
            (qx, qy, qz, qw),
            rospy.Time.now(),
            "fsds/FSCar",
            "fsds/map"
        )

    def speed_callback(self, msg):
       xv = msg.twist.twist.linear.x
       yv = msg.twist.twist.linear.y
       self.state[3] = np.hypot(xv, yv)

    def track_callback(self, msg):
        self.cone_markers = self.track_to_markers(msg)

        self.blue_cones = []
        self.yellow_cones = []
        for cone in msg.track:
            if cone.color == Cone.BLUE:
                self.blue_cones.append(cone)
            elif cone.color == Cone.YELLOW:
                self.yellow_cones.append(cone)

        # 경로 생성 및 Boundary 시각화
        self.calculate_path_independent()
        
        # 경로 스무딩 (선택 사항)
        # self.smooth_path()

    # --------------------------------------------------------------------------
    # Path Planning & Debug Visualization
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # [FIX] Path Planning: Matching Algorithm
    # --------------------------------------------------------------------------
    def calculate_path_independent(self):
        # 1. [NEW] 차량 상태 가져오기 (필터링용)
        cx = self.state[0]
        cy = self.state[1]
        cyaw = np.radians(self.state[2]) # degree to radian

        # 2. [NEW] 로컬 좌표계 변환 및 후방 콘 필터링 함수
        def filter_rear_cones(cones, margin=-2.0):
            """
            margin: 차량 기준 뒤쪽 몇 미터까지 살려둘 것인가? 
                    (예: -2.0이면 내 뒤 2m까진 포함, 그보다 뒤는 삭제)
            """
            filtered = []
            for c in cones:
                dx = c.location.x - cx
                dy = c.location.y - cy
                
                # 회전 행렬로 Local X (차량 앞뒤 거리) 계산
                local_x = dx * np.cos(-cyaw) - dy * np.sin(-cyaw)
                
                if local_x > margin:
                    filtered.append(c)
            return filtered

        # 3. [NEW] 필터링 적용 (이 리스트로 경로 생성 시작)
        valid_blue = filter_rear_cones(self.blue_cones)
        valid_yellow = filter_rear_cones(self.yellow_cones)

        # (기존 로직에서 변수명만 변경: self.blue_cones -> valid_blue)
        bp = [(c.location.x, c.location.y) for c in valid_blue]
        yp = [(c.location.x, c.location.y) for c in valid_yellow]

        if len(bp) < 2 or len(yp) < 2:
            self.mid_points = []
            return

        # 1. 기준이 될 쪽(더 많은 콘이 있는 쪽)과 반대쪽을 정합니다.
        if len(bp) > len(yp):
            primary = np.array(bp)
            secondary = np.array(yp)
        else:
            primary = np.array(yp)
            secondary = np.array(bp)

        # 2. 기준 쪽(Primary)을 차량과 가까운 순서대로 정렬합니다 (Greedy Sort)
        #    (이전에 드린 거리 제한 로직 포함)
        car_pos = np.array([self.state[0], self.state[1]])
        
        primary_sorted = []
        curr = car_pos
        p_copy = primary.tolist()

        while p_copy:
            # 현재 위치(curr)에서 가장 가까운 콘 찾기
            dists = cdist([curr], p_copy)[0]
            idx = np.argmin(dists)
            min_dist = dists[idx]

            # 너무 멀면(트랙 건너편 등) 연결 끊기
            if len(primary_sorted) > 0 and min_dist > MAX_CONE_CONNECT_DIST:
                break
            
            # 첫 번째 점은 차에서 너무 멀면(예: 10m) 시작도 하지 않음 (이상한 점 연결 방지)
            if len(primary_sorted) == 0 and min_dist > 15.0:
                break

            closest = p_copy.pop(idx)
            primary_sorted.append(closest)
            curr = closest

        primary_sorted = np.array(primary_sorted)

        # 3. 정렬된 Primary 콘 하나하나에 대해, 가장 가까운 Secondary 콘을 찾습니다.
        raw_midpoints = []
        TRACK_WIDTH_MAX = 6.0  # [Tuning] 트랙 폭보다 약간 넓게 설정 (이 이상 멀면 짝꿍 아님)

        if len(secondary) > 0:
            for p_pt in primary_sorted:
                # p_pt와 모든 secondary 점들 사이의 거리 계산
                dists = cdist([p_pt], secondary)[0]
                min_idx = np.argmin(dists)
                min_val = dists[min_idx]

                # [핵심] 짝꿍 콘이 트랙 폭 이내에 있을 때만 중점 생성
                if min_val < TRACK_WIDTH_MAX:
                    s_pt = secondary[min_idx]
                    mid_pt = (p_pt + s_pt) / 2
                    raw_midpoints.append(mid_pt)
                else:
                    # 짝꿍이 없으면 Primary 콘에서 트랙 안쪽으로 일정 거리 띄운 가상의 점을 사용하거나
                    # 단순히 그 점은 경로 생성에서 제외합니다.
                    # 여기서는 간단히 제외합니다.
                    pass
        
        # 4. 구해진 중점들(raw_midpoints)을 스플라인으로 부드럽게 연결
        self.mid_points = []
        if len(raw_midpoints) < 2:
            # 중점이 너무 적으면 경로 생성 실패
            return

        try:
            raw_midpoints = np.array(raw_midpoints)
            
            # 중복된 점 제거 (스플라인 에러 방지)
            # 연속된 점들의 거리가 너무 가까우면(0.1m 미만) 제거
            unique_mids = [raw_midpoints[0]]
            for i in range(1, len(raw_midpoints)):
                if np.linalg.norm(raw_midpoints[i] - unique_mids[-1]) > 0.5:
                    unique_mids.append(raw_midpoints[i])
            unique_mids = np.array(unique_mids)

            if len(unique_mids) < 2:
                return

            # 스플라인 생성
            k_val = min(3, len(unique_mids) - 1)
            tck, u = splprep(unique_mids.T, k=k_val, s=0.5) # s는 스무딩 계수
            
            # 경로 점 생성
            u_new = np.linspace(0, 1, len(unique_mids) * 5) # 점 밀도 높임
            x_new, y_new = splev(u_new, tck)

            for i in range(len(x_new)):
                p = Point()
                p.x, p.y = x_new[i], y_new[i]
                self.mid_points.append(p)
                
        except Exception as e:
            rospy.logwarn(f"Spline generation failed: {e}")
            self.mid_points = []   

    def publish_boundary_line(self, sorted_points, color_type):
        """ 정렬된 콘들을 잇는 선을 시각화 """
        marker = Marker()
        marker.header.frame_id = "fsds/map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = f"{color_type}_boundary"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1 # 선 두께

        if color_type == "blue":
            marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
        else:
            marker.color = ColorRGBA(1.0, 1.0, 0.0, 1.0)

        for pt in sorted_points:
            p = Point()
            p.x = pt[0]
            p.y = pt[1]
            p.z = 0.0
            marker.points.append(p)

        if color_type == "blue":
            self.blue_bound_pub.publish(marker)
        else:
            self.yellow_bound_pub.publish(marker)

    def publish_midpoints(self):
        path_msg = Path()
        path_msg.header.frame_id = "fsds/map" 
        path_msg.header.stamp = rospy.Time.now()

        for pt in self.mid_points:
            pose = PoseStamped()
            pose.header.frame_id = "fsds/map"
            pose.pose.position = pt
            pose.pose.orientation.w = 1.0 
            path_msg.poses.append(pose)

        self.midline_path_pub.publish(path_msg)

    # --------------------------------------------------------------------------
    # Control & Dashboard
    # --------------------------------------------------------------------------
    def publish_debug_dashboard(self, target_v, throttle, brake, steering, max_curvature):
        """ 차량 위에 현재 상태를 텍스트와 화살표로 표시 """
        
        # 1. Text Info (HUD)
        text_marker = Marker()
        text_marker.header.frame_id = "fsds/FSCar" # 차 기준
        text_marker.ns = "dashboard"
        text_marker.id = 0
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.scale.z = 0.4 # 글자 크기
        text_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        text_marker.pose.position.z = 1.5 # 차 머리 위 1.5m

        cur_v = self.state[3]
        text_marker.text = (
            f"Speed: {cur_v:.1f} / {target_v:.1f} m/s\n"
            f"Thr: {throttle:.2f} | Brk: {brake:.2f}\n"
            f"Steer: {steering:.2f}\n"
            f"MaxCurv: {max_curvature:.3f}"
        )
        self.text_info_pub.publish(text_marker)

        # 2. Steering Arrow (조향 방향 시각화)
        arrow = Marker()
        arrow.header.frame_id = "fsds/FSCar"
        arrow.ns = "steering"
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.scale = Vector3(0.1, 0.2, 0.2) # 샤프트 두께, 헤드 지름, 헤드 길이
        arrow.color = ColorRGBA(1.0, 0.0, 1.0, 1.0) # Magenta

        # 시작점 (차 중심)
        p_start = Point(0, 0, 0.5)
        
        # 끝점 (조향각 반영)
        # steering 값(-1.0 ~ 1.0)은 대략 30도(0.52rad) 정도라고 가정하고 시각화
        steer_rad = steering * 0.5 # 시각화용 스케일링
        arrow_len = 2.0
        p_end = Point(
            arrow_len * np.cos(steer_rad),
            arrow_len * np.sin(steer_rad),
            0.5
        )
        
        arrow.points = [p_start, p_end]
        self.steering_arrow_pub.publish(arrow)

    def publish_target_marker(self, target_point):
        marker = Marker()
        marker.header.frame_id = "fsds/map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "target_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = target_point
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(0.5, 0.5, 0.5)
        marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0) # Green
        self.target_pub.publish(marker)

    def calculate_path_curvature(self, p1, p2, p3):
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y
        a = np.hypot(x1 - x2, y1 - y2)
        b = np.hypot(x2 - x3, y2 - y3)
        c = np.hypot(x3 - x1, y3 - y1)
        if (a * b * c) == 0: return 0.0
        s = (a + b + c) / 2.0
        area = np.sqrt(abs(s * (s - a) * (s - b) * (s - c)))
        return (4 * area) / (a * b * c)

    def pure_pursuit(self):
        x, y, theta, v = self.state
        yaw_rad = np.radians(theta)
        
        if not self.mid_points: # 경로가 없으면 정지
            return 0.0, 0.0, 1.0

        adaptive_lookahead = LOOKAHEAD_MIN + (LOOKAHEAD_GAIN * v)
        
        # [핵심 수정] 모든 경로점들을 내 차 기준(Local Frame)으로 변환
        # 내 앞(Local X > 0)에 있는 점들 중에서 가장 가까운 점을 찾음
        
        best_target = None
        min_diff = float('inf')

        # 후보군 필터링
        valid_points = []
        
        for pt in self.mid_points:
            # 전역 좌표계에서의 차이
            dx = pt.x - x
            dy = pt.y - y
            
            # 회전 행렬을 이용해 차량 기준 좌표(local_x, local_y)로 변환
            local_x = dx * np.cos(-yaw_rad) - dy * np.sin(-yaw_rad)
            local_y = dx * np.sin(-yaw_rad) + dy * np.cos(-yaw_rad)
            
            # 내 뒤(-0.5m 여유)에 있는 점은 아예 무시 & 너무 먼 점도 무시
            if local_x > -0.5 and np.hypot(dx, dy) < 20.0:
                valid_points.append((pt, local_x, local_y, np.hypot(dx, dy)))

        if not valid_points:
            # 유효한 점이 없으면 멈춤 (안전을 위해)
            return 0.0, 0.0, 1.0

        # 유효한 점들 중에서 Lookahead 거리와 가장 비슷한 점 찾기
        # (단순히 제일 가까운 점이 아니라, Lookahead 원 위에 있는 점을 찾으려 노력)
        for pt, lx, ly, dist in valid_points:
            dist_diff = abs(dist - adaptive_lookahead)
            if dist_diff < min_diff:
                min_diff = dist_diff
                best_target = pt

        target_point = best_target
        
        # --- 아래부터는 기존 로직과 비슷 ---
        
        # Curvature 계산을 위한 인덱스 찾기 (이 부분은 대략적으로 매칭)
        # (정확한 인덱싱이 어렵다면 현재 타겟 포인트 주변 점들을 탐색해야 함)
        # 여기서는 간단히 생략하거나, valid_points 구조를 개선해서 원본 인덱스를 같이 저장하면 좋습니다.
        # 우선 주행 문제 해결을 위해 Curvature 예측은 0.0으로 둡니다.
        max_curvature = 0.0 

        # Control Logic
        target_dx = target_point.x - x
        target_dy = target_point.y - y
        target_theta = np.arctan2(target_dy, target_dx)
        
        # Heading Error (alpha)
        alpha = (target_theta - yaw_rad + np.pi) % (2 * np.pi) - np.pi

        steering = np.arctan2(2 * WHEELBASE * np.sin(alpha), adaptive_lookahead)
        steering = np.clip(steering, -1.0, 1.0)

        # # 속도 제어 (간소화)
        # # 급격한 조향 시 감속
        # target_v = MAX_SPEED
        # if abs(steering) > STEERING_THRESHOLD:
        #     target_v = MIN_SPEED

        # [UPGRADE] 동적 속도 제어 (Dynamic Speed Profile)
        # 조향각(steering)이 클수록 목표 속도를 줄입니다.
        # CORNER_STIFFNESS가 클수록 코너에서 더 많이 감속합니다.
        
        # 1. 조향각의 크기 (절댓값)
        steer_magnitude = abs(steering)
        
        # 2. 감속 적용 (기본 속도에서 조향각 비례 감속)
        # 공식: 목표속도 = 최대속도 / (1.0 + 강성계수 * 조향각)
        target_v = MAX_SPEED / (1.0 + CORNER_STIFFNESS * steer_magnitude)
        
        # 3. 최소 속도 보장 (너무 느려지지 않게)
        target_v = max(target_v, MIN_SPEED)
        
        # ------------------------------------------------------
        
        throttle = 0.0
        brake = 0.0
        speed_error = target_v - v
        
        if speed_error > 0:
            throttle = np.clip(K_ACCEL * speed_error, 0.0, MAX_ACCEL)
        else:
            if speed_error < -0.5:
                brake = np.clip(K_BRAKE * abs(speed_error), 0.0, MAX_BRAKE)

        # [VISUALIZATION]
        self.publish_target_marker(target_point)
        self.publish_debug_dashboard(target_v, throttle, brake, steering, max_curvature)

        return throttle, steering, brake
        
    def run(self):
        x, y, theta, v = self.state
        path = self.mid_points

        # --------------------------------------------------------------------------


        throttle, steering, brake = self.pure_pursuit()

        # --------------------------------------------------------------------------

        cmd = ControlCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.throttle = throttle
        cmd.steering = steering
        cmd.brake = brake
        self.cmd_pub.publish(cmd)

        self.publish_midpoints()
        self.cones_pub.publish(self.cone_markers)

def main():
    rospy.init_node("control_node")
    node = ControlNode()

    rate = rospy.Rate(30) # 30 Hz
    while not rospy.is_shutdown():
        node.run()
        rate.sleep()

if __name__ == "__main__":
    main()
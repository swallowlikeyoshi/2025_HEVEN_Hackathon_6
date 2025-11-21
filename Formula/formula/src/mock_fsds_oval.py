#!/usr/bin/env python3
import rospy
import tf
import numpy as np
from fs_msgs.msg import Track, Cone, ControlCommand
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, TwistWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray

class MockFSDSSimulator:
    def __init__(self):
        rospy.init_node('mock_fsds_simulator')

        # ======================================================================
        # 1. 가상 트랙 생성 (정밀한 간격의 타원형)
        # ======================================================================
        self.track_msg = Track()
        self.generate_oval_track()

        # ======================================================================
        # 2. 차량 상태 초기화
        # ======================================================================
        # 타원 트랙의 동쪽 끝(시작점)에서 북쪽(+y)을 보고 시작
        self.x = 0.0 
        self.y = 0.0
        self.yaw = 0.0
        self.reset_vehicle_position()

        self.v = 0.0
        self.wheelbase = 1.55

        # 제어 입력값 저장
        self.cmd_throttle = 0.0
        self.cmd_brake = 0.0
        self.cmd_steering = 0.0

        # ======================================================================
        # 3. ROS 통신 설정
        # ======================================================================
        # Publisher
        self.pub_track = rospy.Publisher("/fsds/testing_only/track", Track, queue_size=1, latch=True)
        self.pub_odom = rospy.Publisher("/fsds/testing_only/odom", Odometry, queue_size=10)
        self.pub_gss = rospy.Publisher("/fsds/gss", TwistWithCovarianceStamped, queue_size=10)
        self.tf_br = tf.TransformBroadcaster()

        # Subscriber (Control Node에서 오는 명령 수신)
        rospy.Subscriber("/fsds/control_command", ControlCommand, self.cmd_callback)

        # 주기적 실행 (50Hz)
        self.rate = rospy.Rate(50)

    def reset_vehicle_position(self):
        # generate_oval_track 실행 후 호출되어 정확한 위치를 잡음
        # 기본값은 원점이지만, 트랙 생성 함수에서 덮어씌워짐
        pass

    def generate_oval_track(self):
        """
        타원형 트랙 생성
        - 특징: 곡률에 상관없이 콘 간격을 일정하게(약 3.5m) 유지함
        """
        center_x, center_y = 20.0, 10.0
        a, b = 25.0, 15.0  # 장축(25m), 단축(15m)
        track_width = 4.0
        
        # [설정] 콘과 콘 사이의 목표 거리 (단위: m)
        cone_spacing = 3.5 

        curr_theta = 0.0
        
        # 한 바퀴(2pi) 돌 때까지 반복
        while curr_theta < 2 * np.pi:
            # 1. 현재 각도(theta)에서의 중심선 좌표 (x, y)
            cx = center_x + a * np.cos(curr_theta)
            cy = center_y + b * np.sin(curr_theta)
            
            # 2. 법선 벡터(Normal Vector) 계산 -> 트랙 폭만큼 벌리기 위해
            # 타원 접선 벡터: (-a*sin, b*cos)
            # 타원 법선 벡터(수직, 바깥쪽): (b*cos, a*sin)
            nx = b * np.cos(curr_theta)
            ny = a * np.sin(curr_theta)
            
            # 단위 벡터로 정규화
            n_len = np.hypot(nx, ny)
            nx /= n_len
            ny /= n_len

            # 3. 콘 생성 (Blue는 안쪽, Yellow는 바깥쪽)
            # FSDS 기준: 진행방향 왼쪽=Blue, 오른쪽=Yellow
            # 반시계 방향(CCW) 주행 시 안쪽이 Blue
            
            # Blue Cone (Inner) - 법선 반대 방향
            bc = Cone()
            bc.location.x = cx - (nx * track_width / 2.0)
            bc.location.y = cy - (ny * track_width / 2.0)
            bc.color = Cone.BLUE
            self.track_msg.track.append(bc)

            # Yellow Cone (Outer) - 법선 방향
            yc = Cone()
            yc.location.x = cx + (nx * track_width / 2.0)
            yc.location.y = cy + (ny * track_width / 2.0)
            yc.color = Cone.YELLOW
            self.track_msg.track.append(yc)

            # 4. 다음 콘을 찍을 각도(delta_theta) 계산 (미분 이용)
            # 호의 길이 공식: ds = sqrt((dx/dt)^2 + (dy/dt)^2) * dt
            # 따라서 dt(각도변화량) = 목표거리 / sqrt(...)
            ds_dt = np.sqrt((a * np.sin(curr_theta))**2 + (b * np.cos(curr_theta))**2)
            
            delta_theta = cone_spacing / ds_dt
            curr_theta += delta_theta

        # 차량 시작 위치 설정 (트랙 동쪽 끝에서 +y방향을 보고 시작)
        self.x = center_x + a
        self.y = center_y
        self.yaw = np.pi / 2 
        rospy.loginfo(f"Track Generated. Start Pose: ({self.x:.1f}, {self.y:.1f})")

    def cmd_callback(self, msg):
        self.cmd_throttle = msg.throttle
        self.cmd_brake = msg.brake
        self.cmd_steering = msg.steering

    def update_physics(self, dt):
        """
        간이 차량 물리 모델 (Kinematic Bicycle Model)
        주의: 실제 물리 엔진(마찰력, 관성 등)은 포함되지 않았습니다.
        알고리즘 논리 검증용으로만 사용하세요.
        """
        # 가속도 모델 (단순화)
        accel = self.cmd_throttle * 5.0 - self.cmd_brake * 8.0
        
        # 공기 저항 및 마찰
        drag = 0.1 * self.v 
        accel -= drag

        # 속도 업데이트
        self.v += accel * dt
        if self.v < 0: self.v = 0 

        # 위치 업데이트 (Bicycle Model)
        # 조향각 제한을 두면 더 현실적일 수 있음 (옵션)
        steering_angle = np.clip(self.cmd_steering, -1.0, 1.0) # 라디안 가정 시 약 57도 (FSDS 맥스)
        
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += (self.v / self.wheelbase) * np.tan(steering_angle) * dt

    def publish_state(self):
        current_time = rospy.Time.now()

        # 1. Odom Msg
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "fsds/map"
        odom.child_frame_id = "fsds/FSCar"
        
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        q = tf.transformations.quaternion_from_euler(0, 0, self.yaw)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        self.pub_odom.publish(odom)

        # 2. TF Broadcast (Simulation에서는 시뮬레이터가 TF를 뿌려줌)
        self.tf_br.sendTransform(
            (self.x, self.y, 0.0),
            q,
            current_time,
            "fsds/FSCar",
            "fsds/map"
        )

        # 3. Speed Msg (GSS)
        gss = TwistWithCovarianceStamped()
        gss.header.stamp = current_time
        gss.twist.twist.linear.x = self.v
        self.pub_gss.publish(gss)

        # 4. Track Msg (주기적으로 발행, Latch되어 있으므로 1번만 가도 됨)
        self.pub_track.publish(self.track_msg)

    def run(self):
        last_time = rospy.Time.now()
        rospy.loginfo("Mock Simulator Started. Waiting for control commands...")
        
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            dt = (current_time - last_time).to_sec()
            last_time = current_time

            if dt > 0:
                self.update_physics(dt)
                self.publish_state()
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        sim = MockFSDSSimulator()
        sim.run()
    except rospy.ROSInterruptException:
        pass
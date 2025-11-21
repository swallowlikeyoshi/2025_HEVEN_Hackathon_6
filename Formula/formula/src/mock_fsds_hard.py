#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import math
from fs_msgs.msg import Track, Cone, ControlCommand
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, TwistWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray

# ==========================================
# 트랙 생성 빌더 클래스 (블록 조립식)
# ==========================================
class TrackBuilder:
    def __init__(self, track_msg, start_x=0, start_y=0, start_yaw=0, width=4.0, cone_step=2.0):
        self.track_msg = track_msg
        self.x = start_x
        self.y = start_y
        self.yaw = start_yaw
        self.width = width
        self.cone_step = cone_step # 콘 간격 (m)

    def _add_cones_at_curr_pos(self):
        # 왼쪽(Blue), 오른쪽(Yellow) 좌표 계산
        # FSDS 규정: Blue=Left, Yellow=Right
        lx = self.x + (self.width / 2.0) * np.cos(self.yaw + np.pi/2)
        ly = self.y + (self.width / 2.0) * np.sin(self.yaw + np.pi/2)
        
        rx = self.x + (self.width / 2.0) * np.cos(self.yaw - np.pi/2)
        ry = self.y + (self.width / 2.0) * np.sin(self.yaw - np.pi/2)

        # Blue Cone
        bc = Cone()
        bc.location.x = lx
        bc.location.y = ly
        bc.location.z = 0
        bc.color = Cone.BLUE
        self.track_msg.track.append(bc)

        # Yellow Cone
        yc = Cone()
        yc.location.x = rx
        yc.location.y = ry
        yc.location.z = 0
        yc.color = Cone.YELLOW
        self.track_msg.track.append(yc)

    def add_straight(self, length):
        """직선 구간 추가"""
        steps = int(length / self.cone_step)
        for _ in range(steps):
            self.x += self.cone_step * np.cos(self.yaw)
            self.y += self.cone_step * np.sin(self.yaw)
            self._add_cones_at_curr_pos()

    def add_turn(self, radius, angle_deg, direction='left'):
        """곡선 구간 추가 (direction: 'left' or 'right')"""
        angle_rad = np.radians(angle_deg)
        arc_length = radius * angle_rad
        steps = int(arc_length / self.cone_step)
        
        step_angle = angle_rad / steps
        
        for _ in range(steps):
            if direction == 'left':
                self.yaw += step_angle
            else:
                self.yaw -= step_angle
            
            self.x += self.cone_step * np.cos(self.yaw)
            self.y += self.cone_step * np.sin(self.yaw)
            self._add_cones_at_curr_pos()

# ==========================================
# 시뮬레이터 메인 클래스
# ==========================================
class MockFSDSSimulator:
    def __init__(self):
        rospy.init_node('mock_fsds_simulator')

        # 1. 트랙 생성
        self.track_msg = Track()
        # 여기서 원하는 트랙 모양을 선택하세요
        self.generate_complex_endurance_track() 
        # self.generate_skidpad_track() 

        # 2. 차량 상태 초기화
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.wheelbase = 1.55

        # 트랙의 시작점에 차량 배치
        self.reset_vehicle_to_start()

        # 제어 입력값
        self.cmd_throttle = 0.0
        self.cmd_brake = 0.0
        self.cmd_steering = 0.0

        # 3. ROS 통신
        self.pub_track = rospy.Publisher("/fsds/testing_only/track", Track, queue_size=1, latch=True)
        self.pub_odom = rospy.Publisher("/fsds/testing_only/odom", Odometry, queue_size=10)
        self.pub_gss = rospy.Publisher("/fsds/gss", TwistWithCovarianceStamped, queue_size=10)
        self.tf_br = tf.TransformBroadcaster()
        
        rospy.Subscriber("/fsds/control_command", ControlCommand, self.cmd_callback)

        self.rate = rospy.Rate(50) # 50Hz

    def reset_vehicle_to_start(self):
        """트랙 시작점(0,0)에서 약간 앞, 트랙 방향으로 초기화"""
        # 복합 트랙의 시작점은 (0,0)이고 yaw=0(East)입니다.
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0

    def generate_complex_endurance_track(self):
        """
        [Endurance Test Track]
        직선 -> 완만한 코너 -> 급격한 헤어핀(U-Turn) -> 시케인(S자) -> 복귀
        알고리즘의 'Short Circuit(경로 가로지름)' 문제를 테스트하기 최적화됨.
        """
        builder = TrackBuilder(self.track_msg, start_x=0, start_y=0, start_yaw=0, width=4.5, cone_step=1.5)
        
        # 1. 스타트 직선 (가속 테스트)
        builder.add_straight(30.0) 
        
        # 2. 완만한 좌회전
        builder.add_turn(radius=15.0, angle_deg=90, direction='left')
        
        # 3. 짧은 직선
        builder.add_straight(10.0)
        
        # 4. [핵심 테스트] 우측 헤어핀 (반경이 좁음 -> 벡터 정렬 로직 검증용)
        # 여기서 알고리즘이 뒤쪽 콘을 잡으면 실패함
        builder.add_turn(radius=5.0, angle_deg=180, direction='right')
        
        # 5. 직선 (돌아오는 길)
        builder.add_straight(20.0)
        
        # 6. 시케인 (S자 코너 - 조향 응답성 테스트)
        builder.add_turn(radius=8.0, angle_deg=45, direction='left')
        builder.add_straight(5.0)
        builder.add_turn(radius=8.0, angle_deg=45, direction='right')
        
        # 7. 다시 원점으로 돌아오기 위한 큰 코너
        # (완벽하게 닫힌 루프를 만들기는 어렵지만 테스트엔 충분함)
        builder.add_turn(radius=15.0, angle_deg=90, direction='left')
        builder.add_straight(20.0) # Finish line 근처

    def generate_skidpad_track(self):
        """Skidpad (8자 주행)"""
        center_offset = 9.125 + 1.5 # 8자 중심 거리
        radius = 9.125 # 내경
        width = 3.0
        
        # 간단한 원 두개 생성 (하드코딩)
        # ... (생략, 필요하면 추가 작성 가능)

    def cmd_callback(self, msg):
        self.cmd_throttle = msg.throttle
        self.cmd_brake = msg.brake
        self.cmd_steering = msg.steering

    def update_physics(self, dt):
        # 물리 모델 (가속도, 마찰 등)
        accel = self.cmd_throttle * 5.0 - self.cmd_brake * 10.0
        drag = 0.2 * self.v 
        accel -= drag

        self.v += accel * dt
        if self.v < 0: self.v = 0 

        # Bicycle Model Kinematics
        # 후륜 구동 기준
        beta = np.arctan(0.5 * np.tan(self.cmd_steering)) # Slip angle approximation
        
        self.x += self.v * np.cos(self.yaw + beta) * dt
        self.y += self.v * np.sin(self.yaw + beta) * dt
        self.yaw += (self.v / self.wheelbase) * np.sin(beta) * dt # Simplified yaw rate

    def publish_state(self):
        current_time = rospy.Time.now()

        # Odom
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "fsds/map"
        odom.child_frame_id = "fsds/FSCar"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        
        q = tf.transformations.quaternion_from_euler(0, 0, self.yaw)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        self.pub_odom.publish(odom)

        # TF
        self.tf_br.sendTransform((self.x, self.y, 0), q, current_time, "fsds/FSCar", "fsds/map")

        # GSS
        gss = TwistWithCovarianceStamped()
        gss.header.stamp = current_time
        gss.twist.twist.linear.x = self.v
        self.pub_gss.publish(gss)

        # Track (주기적으로)
        self.pub_track.publish(self.track_msg)

    def run(self):
        last_time = rospy.Time.now()
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
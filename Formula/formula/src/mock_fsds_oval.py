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

        # ===========================
        # 1. 가상 트랙 생성 (타원형)
        # ===========================
        self.track_msg = Track()
        self.generate_oval_track()

        # ===========================
        # 2. 차량 상태 초기화
        # ===========================
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.wheelbase = 1.55

        # 제어 입력값 저장
        self.cmd_throttle = 0.0
        self.cmd_brake = 0.0
        self.cmd_steering = 0.0

        # ===========================
        # 3. ROS 통신 설정
        # ===========================
        # Publisher
        self.pub_track = rospy.Publisher("/fsds/testing_only/track", Track, queue_size=1, latch=True)
        self.pub_odom = rospy.Publisher("/fsds/testing_only/odom", Odometry, queue_size=10)
        self.pub_gss = rospy.Publisher("/fsds/gss", TwistWithCovarianceStamped, queue_size=10)
        self.tf_br = tf.TransformBroadcaster()

        # Subscriber (Control Node에서 오는 명령 수신)
        rospy.Subscriber("/fsds/control_command", ControlCommand, self.cmd_callback)

        # 주기적 실행 (50Hz)
        self.rate = rospy.Rate(50)

    def generate_oval_track(self):
        """간단한 타원형 트랙 생성"""
        center_x, center_y = 20.0, 10.0
        a, b = 25.0, 15.0  # 장축, 단축 반경
        track_width = 4.0

        num_cones = 60
        for i in range(num_cones):
            theta = 2 * np.pi * i / num_cones
            
            # Blue Cones (Inner)
            bc = Cone()
            bc.location.x = center_x + (a - track_width/2) * np.cos(theta)
            bc.location.y = center_y + (b - track_width/2) * np.sin(theta)
            bc.color = Cone.BLUE
            self.track_msg.track.append(bc)

            # Yellow Cones (Outer)
            yc = Cone()
            yc.location.x = center_x + (a + track_width/2) * np.cos(theta)
            yc.location.y = center_y + (b + track_width/2) * np.sin(theta)
            yc.color = Cone.YELLOW
            self.track_msg.track.append(yc)

        # 차량 시작 위치를 트랙 중간으로 설정
        self.x = center_x + a
        self.y = center_y
        self.yaw = np.pi / 2  # +y 방향 보고 시작

    def cmd_callback(self, msg):
        self.cmd_throttle = msg.throttle
        self.cmd_brake = msg.brake
        self.cmd_steering = msg.steering

    def update_physics(self, dt):
        """간이 차량 물리 모델 (Kinematic Bicycle Model)"""
        # 가속도 모델 (단순화: Throttle * 5.0 m/s^2, Brake * 8.0 m/s^2)
        accel = self.cmd_throttle * 5.0 - self.cmd_brake * 8.0
        
        # 공기 저항 및 마찰 (속도에 비례해서 감속)
        drag = 0.1 * self.v 
        accel -= drag

        # 속도 업데이트
        self.v += accel * dt
        if self.v < 0: self.v = 0  # 후진 불가 가정

        # 위치 업데이트
        # x_dot = v * cos(yaw)
        # y_dot = v * sin(yaw)
        # yaw_dot = (v / L) * tan(steering)
        
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += (self.v / self.wheelbase) * np.tan(self.cmd_steering) * dt

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

        # 4. Track Msg (주기적으로 발행)
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
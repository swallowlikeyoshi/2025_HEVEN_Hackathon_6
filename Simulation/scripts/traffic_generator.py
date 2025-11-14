#!/usr/bin/env python3

import time
import rospy

from racecar_simulator.msg import Traffic
from std_msgs.msg import Bool

class RandomTrafficGenerator:
    """
    정지선 미션에서 오는 /traffic(traffic, second)을 받아서
    실제 신호등 색(/traffic_random)을 RED/GREEN 으로 바꿔주는 노드.
    동시에, STOP 이후 처음으로 RED -> GREEN 이 되는 순간
    /timer_start 를 True 로 보내서 전체 타이머를 시작시킨다.
    """
    def __init__(self):
        rospy.init_node('traffic_generator', anonymous=True)

        # RViz 신호등용
        self.tf_pub = rospy.Publisher('/traffic_random', Traffic, queue_size=1)
        # 정지선 미션 결과(/traffic) 구독
        self.stop_sub = rospy.Subscriber('/traffic', Traffic, self.traffic_sub)
        # 전체 타이머 시작 신호용
        self.timer_pub = rospy.Publisher('/timer_start', Bool, queue_size=1)

        self.rate = rospy.Rate(100)

        # /traffic 에서 넘어오는 값들
        self.stop = ""          # "STOP" / "None" / 기타
        self.stop_time = 0.0    # 남은 시간(second)

        # 내부 상태
        self.prev_light = "None"          # 이전 신호등 상태 ("RED"/"GREEN")
        self.global_timer_started = False # 전체 타이머 이미 시작했는지
        self.has_seen_stop_phase = False  # STOP 구간(빨간불)을 한 번이라도 겪었는지

    def traffic_sub(self, data):
        self.stop = data.traffic
        self.stop_time = data.second
        

    def main(self):
        traffic_msg = Traffic()

        while not rospy.is_shutdown():
            # 1) /traffic 에서 STOP 이 들어오면: 정지선 미션 진행 중
            if self.stop == "STOP":
                self.has_seen_stop_phase = True

                # 아직 남은 시간이 있으면 빨간불
                if self.stop_time > 0.0:
                    traffic_msg.traffic = "RED"
                # 남은 시간이 0 이하가 되면 초록불
                else:
                    traffic_msg.traffic = "GREEN"

            # 2) STOP 미션이 아닐 때는 마지막 상태 유지 (없으면 GREEN 기본)
            else:
                if self.prev_light in ("RED", "GREEN"):
                    traffic_msg.traffic = self.prev_light
                else:
                    traffic_msg.traffic = "GREEN"

            # 3) "STOP 구간을 최소 한 번 겪고", 바로 직전이 RED였고,
            #    지금 막 GREEN 으로 바뀐 그 순간에만 /timer_start 를 True 로 발행
            if (self.has_seen_stop_phase and
                self.prev_light == "RED" and
                traffic_msg.traffic == "GREEN" and
                not self.global_timer_started):

                start_msg = Bool()
                start_msg.data = True
                self.timer_pub.publish(start_msg)
                self.global_timer_started = True
                rospy.loginfo("[traffic_generator] Start race timer at GREEN")

            # 상태 업데이트 + publish
            self.prev_light = traffic_msg.traffic
            self.tf_pub.publish(traffic_msg)

            self.rate.sleep()


if __name__ == "__main__":
    traffic = RandomTrafficGenerator()
    traffic.main()

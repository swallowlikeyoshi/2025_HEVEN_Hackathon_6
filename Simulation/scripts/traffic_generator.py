#!/usr/bin/env python3

import time
import rospy

from random import randint
from racecar_simulator.msg import Traffic

TRAFFIC = {
    0: "RED",
    1: "GREEN"
}

class RandomTrafficGenerator:
    """
    무작위로 신호등을 바꾸는 클래스
    신호등 색깔은 5초마다 바뀜
    """
    def __init__(self):
        rospy.init_node('traffic_generator', anonymous=True)
        self.tf_pub = rospy.Publisher('/traffic_random', Traffic, queue_size=1)
        self.stop_sub = rospy.Subscriber('/traffic',Traffic, self.traffic_sub)
        self.rate = rospy.Rate(100)  # 100Hz로 설정
        self.current_state = randint(0, 1)  # 0: RED, 1: GREEN
        self.last_change_time = time.time()  # 타이머 초기화
        self.stop = ""
        self.stop_time = 5.0

    def traffic_sub(self, data):
        self.stop = data.traffic
        self.stop_time = data.second
        

    def main(self):
        traffic_msg = Traffic() 
        while not rospy.is_shutdown():
            if self.stop == "STOP":
                traffic_msg.traffic = "RED"
            if self.stop_time == 0.0:
                traffic_msg.traffic = "GREEN"

            self.tf_pub.publish(traffic_msg)
            self.rate.sleep()  

if __name__ == "__main__":
    traffic = RandomTrafficGenerator()
    traffic.main()

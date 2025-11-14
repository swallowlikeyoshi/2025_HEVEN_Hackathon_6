#!/usr/bin/env python3

import rospy
import numpy as np
import time
import random

from goal import *
from racecar_simulator.msg import Complete, Traffic
from parameter_list import Param
from abstract_mission import Mission, Carstatus

param = Param()


class StopLineMission(Mission):
    def __init__(self, map_number) -> None:
        super().__init__()
        self.map_number = map_number

        # Complete는 다시 스폰되는 장소를 업데이트할때만 사용
        self.complete = rospy.Publisher("/stop_complete", Complete, queue_size=1)
        self.traffic = rospy.Publisher("/traffic", Traffic, queue_size=1)

        # For stop mission
        self.stop_time = 0              # 차량이 멈춘 시간
        self.stop_index = 0             # 미션의 현재 상태(state)를 정의하는 변수
        self.stop_flag = 1              # 미션 성공시 1, 실패시 0
        self.stop_duration = param.STOP_LINE_TIME  # 기본값 5초 (fallback)
        if self.map_number == 1:
            self.num_success_stop = [0,0]
        elif self.map_number == 2:
            self.num_success_stop = [0,0,0,0,0]
        elif self.map_number == 3:
            self.num_success_stop = [0,0,0]

    def main(self, goal=Goal, car=Carstatus):
        traffic_msg = Traffic()
        if self.num_success_stop[goal.number - 1] == 1:
            # Already Succeed
            rospy.loginfo("Finished Stop line. Go ahead.")
            traffic_msg.traffic = goal.traffic
            traffic_msg.second = 0
            self.traffic.publish(traffic_msg)
        
        else:
            # --- 정지선 미션 상태머신 ---

            # 0단계: "정지선 근처에 들어온 순간"부터 빨간불 + 랜덤 타이머
            if self.stop_index == 0:
                # 정지선 근처인지(기존 is_in_stopline 그대로 사용)
                if self.is_in_stopline(goal, car):
                    # 처음 들어온 순간에만 랜덤 대기시간 설정
                    if self.stop_time == 0:
                        self.stop_time = time.time()
                        self.stop_duration = random.uniform(3.0, 5.0)
                        rospy.loginfo(
                            f"[STOP_LINE] Approached stop line, RED for {self.stop_duration:.2f} sec"
                        )

                    elapsed = time.time() - self.stop_time

                    if elapsed < self.stop_duration:
                        # 아직 빨간불 유지 구간
                        traffic_msg.traffic = "STOP"
                        traffic_msg.second = max(self.stop_duration - elapsed, 0.0)
                        self.traffic.publish(traffic_msg)
                        return
                    else:
                        # 빨간불 대기시간이 끝난 순간
                        # → STOP + 0 을 한 번 보내고, 다음 state로 넘김
                        traffic_msg.traffic = "STOP"
                        traffic_msg.second = 0.0
                        self.traffic.publish(traffic_msg)

                        self.stop_index = 1   # 이후 단계(성공/실패 판정용)로 이동
                        return

                else:
                    # 아직 정지선 구간에 안 들어왔으면 초기화
                    self.stop_time = 0
                    traffic_msg.traffic = "None"
                    traffic_msg.second = 0.0
                    self.traffic.publish(traffic_msg)
                    return

            elif self.stop_index == 1:
                elapsed = time.time() - self.stop_time

                # After a car stopped
                if elapsed >= self.stop_duration:
                    # Stop success
                    rospy.loginfo("Stop mission success!")
                    self.stop_flag = 1
                    self.stop_index += 1

                elif elapsed < self.stop_duration and abs(car.speed) > 0.00001:
                    # Stop fail
                    rospy.loginfo("Stop mission failed...")
                    self.stop_flag = 0
                    self.stop_index += 1
                
                else:
                    rospy.loginfo("Trying to stop...")

                traffic_msg.traffic = "STOP"
                traffic_msg.second = max(self.stop_duration - elapsed, 0.0)
                self.traffic.publish(traffic_msg)

            elif self.stop_index == 2:
                # stop mission failed, break the function
                if self.stop_flag == 0:
                    self.stop_index = 0

                # stop misison success
                else:
                    self.stop_index += 1
            
            elif self.stop_index == 3:
                # Stop mission finally end (성공했으면, 해당 index가 1로 바뀜)
                self.num_success_stop[goal.number - 1] += self.stop_flag
                
                # Spawn index (change spawn area)
                complete_msg = Complete()
                complete_msg.complete = True
                self.complete.publish(complete_msg)
                
                # Reset the trigger
                self.stop_flag = 0
                self.stop_index = 0
                
    def is_in_mission(self, goal=Goal, car=Carstatus):
        # Check if a car is in the stop line mission
        position_diff = goal.position - car.position
        # We need to check : x, y (close to the area?)
        if abs(position_diff[0]) <= goal.tolerance[0] and abs(position_diff[1]) <= goal.tolerance[1]:
            return True
        
        return False

    def is_in_stopline(self, goal=Goal, car=Carstatus):
        # Check if a car passes stop line
        position_diff = goal.position - car.position
        dist = np.linalg.norm(position_diff)
        # Stop
        # We need to check : x, y (close to the area?), is CAR behind the STOP LINE?
        if np.dot(car.position_unit_vector, -goal.unit_vector)<=0\
            and np.dot(position_diff, -goal.unit_vector)<=0\
            and dist <= 1:

            return True
        
        return False
    
    def publish_null_traffic(self):
        traffic_msg = Traffic()
        traffic_msg.traffic = "None"
        traffic_msg.second = 0
        self.traffic.publish(traffic_msg)

    def init_values(self):
        self.stop_time = 0              # 차량이 멈춘 시간
        self.stop_index = 0             # 미션의 현재 상태(state)를 정의하는 변수
        self.stop_flag = 1              # 미션 성공시 1, 실패시 0
        if self.map_number == 1:
            self.num_success_stop = [0,0]
        elif self.map_number == 2:
            self.num_success_stop = [0,0,0,0,0]
        elif self.map_number == 3:
            pass
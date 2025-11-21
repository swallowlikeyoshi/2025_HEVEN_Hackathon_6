#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math

from database import Database
from ackermann_msgs.msg import AckermannDrive
from parameter_list import Param
from debug import BrainLogger

param = Param()

class Brain():
    def __init__(self, db=Database, map_number=int):
        self.db = db
        
        # --- 차량 제원 및 파라미터 ---
        self.WHEELBASE = 0.325  # 축거 (m)
        self.LIDAR_OFFSET = 0.275 # 오도메트리 기준 전방 오프셋
        
        # --- 튜닝 파라미터 (실전 테스트 필요) ---
        self.LOOKAHEAD_DIST = 0.5  # 전방 주시 거리 (m) - 속도에 따라 가변 가능
        self.MAX_SPEED = 2.0       # 기본 주행 속도
        self.MIN_SPEED = 0.5       # 장애물 감지/코너 시 최저 속도
        self.OBSTACLE_DIST = 0.1   # 전방 장애물 감지 거리 (m)
        self.STOP_DIST = 0.2       # 비상 정지 거리 (m)
        
        self.path_idx = 0          # 최적화: 지난번 찾은 경로 인덱스 저장

    def normalize_angle(self, angle):
        """각도를 -pi ~ pi 사이로 정규화"""
        while angle > np.pi: angle -= 2.0 * np.pi
        while angle < -np.pi: angle += 2.0 * np.pi
        return angle

    def get_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def transform_path_to_local(self, target_x, target_y, current_pose):
        """전역 좌표(target)를 차량 기준 좌표(Local)로 변환"""
        cx, cy, cyaw = current_pose
        
        # 전역 좌표계에서의 차이
        dx = target_x - cx
        dy = target_y - cy
        
        # 회전 변환 (Global -> Local)
        # Local X: 차량 정면, Local Y: 차량 좌측
        lx = dx * np.cos(cyaw) + dy * np.sin(cyaw)
        ly = -dx * np.sin(cyaw) + dy * np.cos(cyaw)
        
        return lx, ly

    def pure_pursuit(self, global_path, current_pose):
        """
        Pure Pursuit 알고리즘
        :return: steering_angle (rad), target_speed_factor (0~1)
        """
        if len(global_path) == 0:
            return 0, 0

        cx, cy, _ = current_pose
        
        # 1. 가장 가까운 경로점 찾기 (이전 인덱스부터 검색하여 연산 속도 확보)
        # path가 순환하지 않는다면 범위 제한 필요, 여기서는 전체 검색 혹은 윈도우 검색
        # 해커톤 맵 크기가 작으므로 전체 검색도 무방하나, 효율을 위해 윈도우잉 추천
        
        # 단순화를 위해 현재 위치에서 모든 점까지의 거리 계산
        path_arr = np.array(global_path)
        dists = np.linalg.norm(path_arr - np.array([cx, cy]), axis=1)
        min_idx = np.argmin(dists)
        self.path_idx = min_idx # 상태 업데이트

        # 2. Lookahead Point 찾기
        target_idx = min_idx
        lookahead_dist = self.LOOKAHEAD_DIST
        
        # 경로 끝까지 탐색하며 Lookahead 거리보다 멀어지는 첫 지점 선택
        for i in range(min_idx, len(global_path)):
            dist = self.get_distance(global_path[i], (cx, cy))
            if dist > lookahead_dist:
                target_idx = i
                break
        
        target_pt = global_path[target_idx]
        
        # 3. 조향각 계산
        # 목표 지점을 차량 기준 좌표계(Local)로 변환
        lx, ly = self.transform_path_to_local(target_pt[0], target_pt[1], current_pose)
        
        # Pure Pursuit 공식: delta = atan(2 * L * sin(alpha) / Ld)
        # 여기서 alpha는 목표 지점의 방위각 -> atan2(ly, lx)
        # lx가 0에 가까우면(바로 옆) 조향을 크게 틀어야 함
        
        lookahead_actual = np.sqrt(lx**2 + ly**2)
        steering_angle = np.arctan2(2.0 * self.WHEELBASE * ly, lookahead_actual**2)
        
        # 4. 곡률 기반 감속 팩터 (직선은 1.0, 급커브는 낮음)
        speed_factor = 1.0
        if abs(steering_angle) > 0.3: # 약 17도 이상 조향 시
            speed_factor = 0.4
        elif abs(steering_angle) > 0.15:
            speed_factor = 0.7
            
        return steering_angle, speed_factor

    def check_lidar_obstacle(self, lidar_data):
        """
        LiDAR 데이터를 이용하여 전방 장애물 확인
        규정: 360포인트, 후방 기준 반시계 방향
        Index 0: 후방 / 90: 우측 / 180: 전방 / 270: 좌측
        """
        if lidar_data is None or len(lidar_data) == 0:
            return False, 1.0

        # 전방 60도 부채꼴 감시 (180도 기준 +/- 30도 -> Index 150 ~ 210)
        # 배열 인덱스 범위 체크
        front_min_idx = 170
        front_max_idx = 190
        
        front_ranges = np.array(lidar_data[front_min_idx:front_max_idx])
        
        # 유효한 데이터만 필터링 (0이나 inf 제외)
        valid_mask = (front_ranges > 0.01) & (front_ranges < 10.0)
        valid_front = front_ranges[valid_mask]
        
        if len(valid_front) == 0:
            return False, 1.0
            
        min_dist = np.min(valid_front) - self.LIDAR_OFFSET # LiDAR 위치 보정
        
        is_emergency = False
        brake_factor = 1.0
        
        if min_dist < self.STOP_DIST:
            # 너무 가까우면 비상 정지
            is_emergency = True
            brake_factor = 0.0
        elif min_dist < self.OBSTACLE_DIST:
            # 장애물이 보이면 거리에 비례해 감속
            # 거리가 멀면 1에 가깝게, 가까우면 0에 가깝게
            brake_factor = (min_dist - self.STOP_DIST) / (self.OBSTACLE_DIST - self.STOP_DIST)
            brake_factor = max(0.2, brake_factor) # 최소 기어가는 속도 유지
            
        return is_emergency, brake_factor

    def main(self):
        # ======================= 데이터 수신 =======================
        lidar_data = self.db.lidar_data
        pose_data = self.db.pose_data # [x, y, yaw(deg)]
        global_path = self.db.global_path
        
        # 안전 장치: 경로 로딩 전 대기
        if len(global_path) == 0:
            print("Waiting for global path...")
            return 0, 0
            
        # Pose 데이터 단위 변환 (Degree -> Radian)
        current_pose = [pose_data[0], pose_data[1], np.radians(pose_data[2])]
        
        # 미션 및 신호등 정보
        curr_mission = self.db.current_mission # 2: 정지선, 1: 주차
        traffic_light = self.db.traffic_light # "RED", "GREEN" (문자열인지 정수인지 확인 필요, 보통 문자열)
        
        # ======================= 1. 주행 제어 (Pure Pursuit) =======================
        target_angle, speed_factor = self.pure_pursuit(global_path, current_pose)
        
        # 기본 속도 설정
        target_speed = self.MAX_SPEED * speed_factor
        
        # ======================= 2. 장애물 감지 (LiDAR) =======================
        is_obstacle, obs_brake_factor = self.check_lidar_obstacle(lidar_data)
        
        # 장애물 감속 적용
        target_speed *= obs_brake_factor
        
        if is_obstacle:
            print("!! OBSTACLE DETECTED !!")
            # 장애물이 있어도 조향은 유지 (회피 시도) 하되 속도는 줄임
            # 만약 완전히 막히면 정지
            if obs_brake_factor == 0.0:
                target_speed = 0
        
        # ======================= 3. 미션 로직 =======================
        
        # [신호등 미션]
        # 미션 ID가 정지선(2)이고 신호가 빨간불이면 정지
        # DB값 확인 필요: traffic_light가 'RED' 스트링인지, 상수인지. 여기선 예시로 처리
        is_red_light = (traffic_light == "RED" or traffic_light == 0) # simulator implementation dependent
        
        if curr_mission == 2 and is_red_light:
            print("Mission: Traffic Light (RED) -> STOP")
            target_speed = 0
            
        # [주차 미션]
        # 주차 미션(1)이면 주차 구역 근처에서 정밀 제어 필요
        # 여기서는 일단 감속하여 진입하도록 설정
        if curr_mission == 1:
            # 주차는 별도 알고리즘(후진 등)이 필요하지만, 
            # 우선은 해당 구역에서 천천히 이동하며 정렬 시도
            target_speed = min(target_speed, 1.0)
            # 주차 상세 로직은 'parking_info'를 활용해 
            # target_angle을 덮어쓰는 방식으로 구현해야 함
            pass

        # 조향각 제한 (-0.5 ~ 0.5 rad) -> 대략 -28도 ~ 28도
        target_angle = np.clip(target_angle, -0.5, 0.5)
        target_speed = np.clip(target_speed, 0, 4.0) # 시뮬레이터 최대 속도 제한

        return target_angle, target_speed

def shutdown_handler(control_pub):
    rospy.loginfo("Shutting down... Publishing stop command.")
    stop_msg = AckermannDrive()  
    stop_msg.speed = 0
    stop_msg.steering_angle = 0
    for _ in range(5): 
        control_pub.publish(stop_msg)
        rospy.sleep(0.1)  

if __name__ == "__main__":
    db = Database(lidar=True)
    test_brain = Brain(db)
    rate = rospy.Rate(param.thread_rate)
    control_pub = rospy.Publisher('/drive', AckermannDrive, queue_size=1)
    logger = BrainLogger(test_brain)

    rospy.on_shutdown(lambda: shutdown_handler(control_pub))

    while not rospy.is_shutdown():
        car_angle, car_speed = test_brain.main()
        motor_msg = AckermannDrive()
        motor_msg.steering_angle = car_angle
        motor_msg.speed = car_speed
        control_pub.publish(motor_msg)
        
        # 디버깅 로그
        # logger.log_data() 
        
        rate.sleep()
#!/usr/bin/env python3
import rospy
import csv
import os
import rospkg
import numpy as np
from nav_msgs.msg import OccupancyGrid

class MovingObstacle:
    def __init__(self):
        rospy.init_node("moving_obstacle")

        #--------------------------------------------------
        # 1) CSV path 설정
        #--------------------------------------------------
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("racecar_simulator")
        self.csv_path = os.path.join(pkg_path, "csv", "obs_move.csv")

        # CSV 로드
        self.path = self.load_csv(self.csv_path)
        self.N = len(self.path)

        if self.N == 0:
            rospy.logerr("CSV path is empty! Cannot run moving obstacle.")
            return

        rospy.loginfo(f"[MovingObstacle] Loaded {self.N} path points.")

        #--------------------------------------------------
        # 2) /map 구독 (원본 map 받아서 저장)
        #--------------------------------------------------
        self.map_received = False
        self.base_map = None
        self.map_width = None
        self.map_height = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.obs_speed = rospy.get_param('~obs_speed', 500)  # default 500Hz
        rospy.Subscriber("/map_static", OccupancyGrid, self.map_callback)
        self.map_pub = rospy.Publisher("/map", OccupancyGrid, queue_size=5)

        #--------------------------------------------------
        # 3) 인덱스 왕복용 변수
        #--------------------------------------------------
        self.idx = 0
        self.direction = 1   # +1 또는 -1

        rate = rospy.Rate(self.obs_speed)  # obs_speed Hz
        while not rospy.is_shutdown():
            if self.map_received:
                self.update()
            rate.sleep()

    #======================================================
    # CSV 로드 함수
    #======================================================
    def load_csv(self, path):
        pts = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader)                 # 헤더 skip
            for row in reader:
                x = float(row[0])
                y = float(row[1])
                pts.append((x, y))
        return pts

    #======================================================
    # /map 수신
    #======================================================
    def map_callback(self, msg):
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        data = np.array(msg.data, dtype=np.int8).reshape(self.map_height, self.map_width)
        self.base_map = data.copy()
        self.map_received = True

    #======================================================
    # 메인 업데이트 (moving obstacle 생성)
    #======================================================
    def update(self):
        # 1) base map 복사
        grid = self.base_map.copy()

        # 2) 현재 CSV 좌표 가져오기
        x, y = self.path[self.idx]

        # 3) world → grid 변환
        gx = int(round((x - self.origin_x) / self.resolution))
        gy = int(round((y - self.origin_y) / self.resolution))

        # 4) grid 범위 안이면 주변 3x3 장애물로 (100)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    grid[ny, nx] = 100

        # 5) OccupancyGrid로 publish
        self.publish_grid(grid)

        # 6) 인덱스 왕복 업데이트
        self.idx += self.direction
        if self.idx >= self.N - 1:
            self.idx = self.N - 1
            self.direction = -1
        elif self.idx <= 0:
            self.idx = 0
            self.direction = 1

    #======================================================
    # Publish OccupancyGrid
    #======================================================
    def publish_grid(self, grid):
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.info.resolution = self.resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        self.map_pub.publish(msg)

#==========================================================
# main
#==========================================================
if __name__ == "__main__":
    try:
        MovingObstacle()
    except rospy.ROSInterruptException:
        pass

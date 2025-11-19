import os
from visualization_msgs.msg import Marker, MarkerArray

class Param():
    def __init__(self):
        # Information of a car
        self.car_angular_velocity = 1
        self.car_acceleration = 1
        self.car_jerk = 0
        self.WHEELBASE = 0.425
        self.REAR_LIDAR = 0.325
        self.WIDTH = 0.145
        self.SIZE_OF_TROPHY = 0.45

        # Endpoint of Map 1
        self.END_POINT_X_1 = 12.5
        self.END_POINT_Y_1 = 8.95

        # Size of stop lane
        self.STOP_LINE_SIZE = 0.8

        # Size of parking lot
        self.PARKING_LOT_WIDTH = 0.55
        self.PARKING_LOT_HEIGHT = 0.8
        
        # MAP 1
        # ===================================================================================
        self.STOP_LINE_TIME = 5

        # Center of stop line
        
        # 25_hackaton stop line location
        self.MAP_1_STOP_LINE_X_1 = 3.0
        self.MAP_1_STOP_LINE_Y_1 = 0.0
        self.MAP_1_STOP_LINE_YAW_1 = 0.0 

        # Random area of parking
        self.MAP_1_PARKING_AREA = 1

        # Center point of parking lot in MAP 1
        #전면 주차
        self.MAP_1_PARKING_LOT_X_1 = 2.548  
        self.MAP_1_PARKING_LOT_Y_1 = 2.743  
        self.MAP_1_PARKING_LOT_YAW_1 = -90  

        #후면 주차
        self.MAP_1_PARKING_LOT_X_2 = 3.736  
        self.MAP_1_PARKING_LOT_Y_2 = 7.249  
        self.MAP_1_PARKING_LOT_YAW_2 = 180  
        
        #평행 주차
        self.MAP_1_PARKING_LOT_X_3 = 9.580  
        self.MAP_1_PARKING_LOT_Y_3 = 7.969  
        self.MAP_1_PARKING_LOT_YAW_3 = 0    

        #주차구역
        self.MAP_1_PARKING_AREA_1_MINX = self.MAP_1_PARKING_LOT_X_1 - 0.3
        self.MAP_1_PARKING_AREA_1_MAXX = self.MAP_1_PARKING_LOT_X_1 + 0.3
        self.MAP_1_PARKING_AREA_1_MINY = self.MAP_1_PARKING_LOT_Y_1 - 0.3
        self.MAP_1_PARKING_AREA_1_MAXY = self.MAP_1_PARKING_LOT_Y_1 + 0.3
        
        self.MAP_1_PARKING_AREA_2_MINX = self.MAP_1_PARKING_LOT_X_2 - 0.3
        self.MAP_1_PARKING_AREA_2_MAXX = self.MAP_1_PARKING_LOT_X_2 + 0.3
        self.MAP_1_PARKING_AREA_2_MINY = self.MAP_1_PARKING_LOT_Y_2 - 0.3
        self.MAP_1_PARKING_AREA_2_MAXY = self.MAP_1_PARKING_LOT_Y_2 + 0.3
        
        self.MAP_1_PARKING_AREA_3_MINX = self.MAP_1_PARKING_LOT_X_3 - 0.3
        self.MAP_1_PARKING_AREA_3_MAXX = self.MAP_1_PARKING_LOT_X_3 + 0.3
        self.MAP_1_PARKING_AREA_3_MINY = self.MAP_1_PARKING_LOT_Y_3 - 0.3
        self.MAP_1_PARKING_AREA_3_MAXY = self.MAP_1_PARKING_LOT_Y_3 + 0.3

        # Tilt degree of parking lot in MAP 1 
        self.PARKING_LOT_TILT_DEGREE_1 = self.MAP_1_PARKING_LOT_YAW_1 + 90
        self.PARKING_LOT_TILT_DEGREE_2 = self.MAP_1_PARKING_LOT_YAW_2 + 90
        self.PARKING_LOT_TILT_DEGREE_3 = self.MAP_1_PARKING_LOT_YAW_3 + 90

        # Obstacle location
        self.MAP_1_OBS_1_x = 4.5
        self.MAP_1_OBS_1_y = 8.62
        self.MAP_1_OBS_2_x = 5.5
        self.MAP_1_OBS_2_y = 9.23


        # Spawn lists ( x, y, yaw(degree) )
        # 각각에 대응하는 GP index    시작     s커브       동적장애물       전면주차            후면주차       정적장애물       평행주차                     
        self.MAP_1_SPAWN_POINT = [(0,0,0), (7.5,0,0), (7.5,4.5,180), (3.5,4.5,180), (1.5,9.0,0), (3.5,9.0,0), (8.0,9.0,0)]

        # Goal Marker
        self.m = Marker()
        self.m.header.frame_id = "map"
        self.m.ns = "goal_marker"
        self.m.type = Marker.LINE_STRIP
        self.m.action = Marker.ADD
        # 정지선 검정색으로 표시 
        self.m.color.r, self.m.color.g, self.m.color.b = 0, 0, 0
        self.m.color.a = 1
        self.m.scale.x = 0.1
        self.m.scale.y = 0.1
        self.m.scale.z = 0

        # Rate of each thread
        self.thread_rate = 10
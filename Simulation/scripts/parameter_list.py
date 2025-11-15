import os
from visualization_msgs.msg import Marker, MarkerArray

# (Map 1) 주차 공간 (1, 2) -> bash 파일로부터 input 받아옴
MAP_1_PARKING_AREA = 1 #int(os.getenv("MAP_1_PARKING_AREA", 1))  # 기본값 1

class Param():
    def __init__(self):
        # Information of a car
        self.car_angular_velocity = 1
        self.car_acceleration = 1
        self.car_jerk = 0
        self.WHEELBASE = 0.425
        self.REAR_LIDAR = 0.325
        self.WIDTH = 0.145
        self.SIZE_OF_TROPHY = 0.5

        # Endpoint of Map 1
        self.END_POINT_X_1 = 33.580
        self.END_POINT_Y_1 = -21.507

        # Size of stop lane
        self.STOP_LINE_SIZE = 0.8

        # Size of parking lot
        self.PARKING_LOT_WIDTH = 0.55
        self.PARKING_LOT_HEIGHT = 0.8
        
        # MAP 1
        # ===================================================================================
        self.STOP_LINE_TIME = 5

        # Center of stop line
        
        # 24_heackaton stop line location
        self.MAP_1_STOP_LINE_X_1 = -6.0
        self.MAP_1_STOP_LINE_Y_1 = -0.0
        self.MAP_1_STOP_LINE_YAW_1 = 180.0 

        self.MAP_1_STOP_LINE_X_2 = 11.698
        self.MAP_1_STOP_LINE_Y_2 = -19.560
        self.MAP_1_STOP_LINE_YAW_2 = -90

        # Check point of left or right road
        self.MAP_1_CHECK_X_1 = -6.02
        self.MAP_1_CHECK_Y_1 = -15.87
        self.MAP_1_CHECK_X_2 = -1.89
        self.MAP_1_CHECK_Y_2 = -15.87

        # Random area of parking
        self.MAP_1_PARKING_AREA = MAP_1_PARKING_AREA

        # Center point of parking lot in MAP 1
        #전면 주차 //테스트 중//
        self.MAP_1_PARKING_LOT_X_1 = 2.548  #2.548
        self.MAP_1_PARKING_LOT_Y_1 = 2.743  #2.743
        self.MAP_1_PARKING_LOT_YAW_1 = -90  #-90

        #후면 주차
        self.MAP_1_PARKING_LOT_X_2 = 3.736  #3.736
        self.MAP_1_PARKING_LOT_Y_2 = 7.249  #7.249
        self.MAP_1_PARKING_LOT_YAW_2 = 180  #180
        
        #평행 주차
        self.MAP_1_PARKING_LOT_X_3 = 9.580  #9.580
        self.MAP_1_PARKING_LOT_Y_3 = 7.969  #7.969
        self.MAP_1_PARKING_LOT_YAW_3 = 0    #0

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
        self.MAP_1_OBS_1_x = 6.18
        self.MAP_1_OBS_1_y = 8.62
        self.MAP_1_OBS_2_x = 6.96
        self.MAP_1_OBS_2_y = 9.23
        self.MAP_1_OBS_yaw = 0.0
        self.MAP_1_OBS_MINX = -7.15
        self.MAP_1_OBS_MINY = 2.3
        self.MAP_1_OBS_MAXX = -0.8
        self.MAP_1_OBS_MAXY = 4.7
        self.MAP_1_OBS_END_MINX = -0.9
        self.MAP_1_OBS_END_MINY = 2.4
        self.MAP_1_OBS_END_MAXX = 1.734
        self.MAP_1_OBS_END_MAXY = 4.97

        # S-curve location
        self.MAP_1_CURVE_MINX = -9.47
        self.MAP_1_CURVE_MINY = 9.5
        self.MAP_1_CURVE_MAXX = -1.05
        self.MAP_1_CURVE_MAXY = 12.91
        self.MAP_1_CURVE_END_MINX = -9.15
        self.MAP_1_CURVE_END_MINY = 10.45
        self.MAP_1_CURVE_END_MAXX = -5.61
        self.MAP_1_CURVE_END_MAXY = 13.19
        self.MAP_1_CURVE_OBS_1_x = -1.2727
        self.MAP_1_CURVE_OBS_1_y = 11.3
        self.MAP_1_CURVE_OBS_2_x = -1.2727
        self.MAP_1_CURVE_OBS_2_y = 12.5
        self.MAP_1_CURVE_OBS_3_x = -3.22
        self.MAP_1_CURVE_OBS_3_y = 9.90
        self.MAP_1_CURVE_OBS_4_x = -3.22
        self.MAP_1_CURVE_OBS_4_y = 11.10
        self.MAP_1_CURVE_OBS_5_x = -2.30
        self.MAP_1_CURVE_OBS_5_y = 11.80
        self.MAP_1_CURVE_OBS_6_x = -2.160
        self.MAP_1_CURVE_OBS_6_y = 10.58
        self.MAP_1_CURVE_OBS_7_x = -4.10
        self.MAP_1_CURVE_OBS_7_y = 12.0
        self.MAP_1_CURVE_OBS_8_x = -4.1
        self.MAP_1_CURVE_OBS_8_y = 10.32

        # G Path drive obs
        self.OBS1 = [-8.472129821777344,-0.501049816608429]
        self.OBS2 = [-7.677031993865967, 0.7769052386283875]    
        self.OBS3 = [-7.5979533195495605, 1.7060810327529907]
        self.OBS4 = [-7.675574779510498, 2.6439454555511475]
        self.OBS5 = [-9.182735443115234, 2.0812268257141113]
        self.OBS6 = [-8.885238647460938, 3.4065401554107666]
        self.OBS7 = [-7.759991645812988, 4.072559356689453]
        self.OBS8 = [-6.545097827911377, 0.8381971716880798]
        self.OBS9 = [-5.517614841461182, 0.7671397924423218]
        self.OBS10 = [-3.372110366821289, 2.271702289581299]
        self.OBS11 = [-4.548928737640381, 2.1779046058654785]
        self.OBS12 = [-6.87177038192749, 12.780014038085938]
        self.OBS13 = [-6.948289394378662, 10.779542922973633]
        self.OBS14 = [-4.712371349334717, 12.772171020507812]
        self.OBS15 = [-4.7899556159973145, 10.954608917236328]
        self.OBS16 = [-9.162704467773438, 1.3111636638641357]
        self.OBS17 = [-9.128958702087402, 0.36640414595603943]
        self.OBS18 = [-7.856508255004883, 12.424349784851074]
        self.OBS19 = [-8.661224365234375, 11.772475242614746]
        self.OBS20 = [-9.017743110656738, 10.865962028503418]
        self.OBS21 = [-5.700079441070557, 11.296499252319336]
        self.OBS22 = [-5.556004047393799, 12.820558547973633]
        self.OBS23 = [-5.734231472015381, 11.19116497039795]


        # Spawn lists ( x, y, yaw(degree) )
        # 각각에 대응하는 GP index    시작     s커브       동적장애물      전면주차     후면주차       정적장애물       평행주차                     
        self.MAP_1_SPAWN_POINT = [(0,0,0), (7.5,0,0), (7.5,4.5,0), (3.0,4.5,0), (1.5,8.5,0), (4.0,8.5,0), (8.0,8.5,0)]
        self.MAP_2_SPAWN_POINT = [(0,0,0), (7.5465,0,0), (11.19,0,0), (21.54,0,0), (35.020,3.675, -90), (25.931,1.044,180), (15.237,1.133,180), (0.1,1.210345,180), (-8.3047,1.2131,180)]
        self.MAP_3_SPAWN_POINT = [(0,0,0), (10.380,-10.700,-87.0894), (10.6453,-19.5551,-87.0895), (10.7380,-25.9737,-88.1929), (-6.865,-27.5207,-178.2), (-15.9061,-45.2137,-94.5), (-15.9783,-47.7374,-94.5)]
        self.MAP_4_SPAWN_POINT = [(0,0,0)]
        self.MAP_5_SPAWN_POINT = [(0,0,0)]

        # Goal Marker
        self.m = Marker()
        self.m.header.frame_id = "map"
        self.m.ns = "goal_marker"
        self.m.type = Marker.CYLINDER
        self.m.action = Marker.ADD
        self.m.color.r, self.m.color.g, self.m.color.b = 1, 1, 0
        self.m.color.a = 1
        self.m.scale.x = 0.5
        self.m.scale.y = 0.5
        self.m.scale.z = 0

        # Rate of each thread
        self.thread_rate = 10
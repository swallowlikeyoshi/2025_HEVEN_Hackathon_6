#!/usr/bin/env python3
import rospy
from parameter_list import Param
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
import numpy as np


class ObstacleCreator:
    def __init__(self):
        rospy.init_node('obstacle_maker')
        
        self.param = Param()

        self.point_sub = rospy.Subscriber('/clicked_point', PointStamped, self.clicked_point_callback)
        self.grid_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=10)

        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)

        self.marker_array = MarkerArray()
        self.marker_id = 0

        self.map_resolution = 0.1
        self.map_width = 100  
        self.map_height = 100  
        self.map_origin_x = -5.0 
        self.map_origin_y = -5.0
        self.grid_data = None

        self.obstacle_data = None
        self.map_received = False

        self.obs1_x = self.param.MAP_1_OBS_1_x
        self.obs1_y = self.param.MAP_1_OBS_1_y
        self.obs2_x = self.param.MAP_1_OBS_2_x
        self.obs2_y = self.param.MAP_1_OBS_2_y
        self.obs_list = [[self.obs1_x, self.obs1_y],
                         [self.obs2_x, self.obs2_y]]

        self.curve_obs_list = [
            [self.param.MAP_1_CURVE_OBS_1_x, self.param.MAP_1_CURVE_OBS_1_y],
            [self.param.MAP_1_CURVE_OBS_2_x, self.param.MAP_1_CURVE_OBS_2_y],
            [self.param.MAP_1_CURVE_OBS_3_x, self.param.MAP_1_CURVE_OBS_3_y],
            [self.param.MAP_1_CURVE_OBS_4_x, self.param.MAP_1_CURVE_OBS_4_y],
            [self.param.MAP_1_CURVE_OBS_5_x, self.param.MAP_1_CURVE_OBS_5_y],
            [self.param.MAP_1_CURVE_OBS_6_x, self.param.MAP_1_CURVE_OBS_6_y],
            [self.param.MAP_1_CURVE_OBS_7_x, self.param.MAP_1_CURVE_OBS_7_y],
            [self.param.MAP_1_CURVE_OBS_8_x, self.param.MAP_1_CURVE_OBS_8_y]
        ]

        self.path_obs_list = [
            self.param.OBS1,
            self.param.OBS2,
            self.param.OBS3,
            self.param.OBS4,
            self.param.OBS5,
            self.param.OBS6,
            self.param.OBS7,
            self.param.OBS8,
            self.param.OBS9,
            self.param.OBS10,
            self.param.OBS11,
            self.param.OBS12,
            self.param.OBS13,
            self.param.OBS14,
            self.param.OBS15,
            self.param.OBS16,
            self.param.OBS17,
            self.param.OBS18,
            self.param.OBS19,
            self.param.OBS20,
            self.param.OBS21,
            self.param.OBS22,
            self.param.OBS23
        ]

        self.rate = rospy.Rate(self.param.thread_rate)

        while not rospy.is_shutdown():
            if self.map_received:
                self.main()
            else:
                rospy.logwarn("Waiting for map to be received...")
            self.rate.sleep()
    
    def clicked_point_callback(self, msg):
        rospy.loginfo("Clicked point received: ({}, {})".format(msg.point.x, msg.point.y))
        self.add_obstacle_to_grid1x1(msg.point.x, msg.point.y)

    def map_callback(self, msg):
        if not self.map_received:
            self.map_width = msg.info.width
            self.map_height = msg.info.height
            self.map_resolution = msg.info.resolution
            self.map_origin_x = msg.info.origin.position.x
            self.map_origin_y = msg.info.origin.position.y

            self.grid_data = np.array(msg.data, dtype=np.int8).reshape(self.map_height, self.map_width)
            self.obstacle_data = np.copy(self.grid_data)

            self.map_received = True
            rospy.loginfo("Map received and initialized.")

    def main(self):
        if not self.map_received:
            rospy.logwarn("Map not received yet. Waiting for map...")
            return

        for x,y in self.obs_list:
            self.add_obstacle_to_grid3x3(x, y)

        for x,y in self.curve_obs_list:
            self.add_obstacle_to_grid1x1(x, y)

        for x,y in self.path_obs_list:
            self.add_obstacle_to_grid1x1(x, y)

        self.publish_grid()

    def add_obstacle_to_grid3x3(self, x, y):
        if self.grid_data is None:
            rospy.logwarn("Grid data is not initialized. Skipping obstacle addition.")
            return

        grid_x = round((x - self.map_origin_x) / self.map_resolution)
        grid_y = round((y - self.map_origin_y) / self.map_resolution)

        for dx in range(-1, 2):  # Adjust to cover a 3x3 square
            for dy in range(-1, 2):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    self.grid_data[ny, nx] = 100

    def add_obstacle_to_grid1x1(self, x, y):
        if self.grid_data is None:
            rospy.logwarn("Grid data is not initialized. Skipping obstacle addition.")
            return

        grid_x = round((x - self.map_origin_x) / self.map_resolution)
        grid_y = round((y - self.map_origin_y) / self.map_resolution)

        if 0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height:
            self.grid_data[grid_y, grid_x] = 100


    def publish_grid(self):
        if self.grid_data is None:
            rospy.logwarn("Grid data is not initialized. Cannot publish grid.")
            return
        updated_map = np.copy(self.grid_data)
        updated_map[self.obstacle_data == 100] = 100

        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id = "map"
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.info.resolution = self.map_resolution
        grid_msg.info.width = self.map_width
        grid_msg.info.height = self.map_height
        grid_msg.info.origin.position.x = self.map_origin_x
        grid_msg.info.origin.position.y = self.map_origin_y
        grid_msg.info.origin.orientation.w = 1.0
        grid_msg.data = updated_map.flatten().astype(int).tolist()  # Ensure correct type

        self.grid_pub.publish(grid_msg)

if __name__ == '__main__':
    try:
        ObstacleCreator()
    except rospy.ROSInterruptException:
        pass

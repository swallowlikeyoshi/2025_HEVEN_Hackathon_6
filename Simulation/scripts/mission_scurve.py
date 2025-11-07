#!/usr/bin/env python3

import rospy
import numpy as np
import time

from goal import *
from racecar_simulator.msg import Complete
from parameter_list import Param
from abstract_mission import Mission, Carstatus

class ScurveMission(Mission):
    def __init__(self, map_number) -> None:
        super().__init__()
        self.map_number = map_number
        self.complete = rospy.Publisher("/scurve_complete", Complete, queue_size=1)
        self.param = Param()
        
        self.num_success_scurve = [0,0]

    def main(self, goal=Goal, car=Carstatus):
        if self.num_success_scurve[goal.number - 1] == 1:
            rospy.loginfo("Finished S-curve. Go ahead.")
            return
        # 장애물 미션 끝나는 지점 근처에 도달하면 그냥 바로 종료   
        if (self.param.MAP_1_CURVE_END_MINX <= car.position[0] <= self.param.MAP_1_CURVE_END_MAXX and
            self.param.MAP_1_CURVE_END_MINY <= car.position[1] <= self.param.MAP_1_CURVE_END_MAXY):
            if self.num_success_scurve[goal.number -1] == 0:
                complete_msg = Complete()
                complete_msg.complete = True
                self.complete.publish(complete_msg)
                rospy.loginfo("S-curve mission success!.")
                self.num_success_scurve[goal.number - 1] = 1
        else :
            rospy.loginfo("Trying curve mission . . . ")
        return    

    def is_in_mission(self, goal=Goal, car=Carstatus):
        # 장애물 등장하는 바운더리 네모구역을 기준으로 mission 구역 판단    
        if (self.param.MAP_1_CURVE_MINX <= car.position[0] <= self.param.MAP_1_CURVE_MAXX and
            self.param.MAP_1_CURVE_MINY <= car.position[1] <= self.param.MAP_1_CURVE_MAXY):
            return True
        return False
    
    def init_values(self):
        self.num_success_scurve = [0,0]
        return

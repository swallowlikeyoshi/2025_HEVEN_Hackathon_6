#!/bin/bash

# 주차구역을 input으로 지정받고 parameter.py의 주차 구역 param을 직접 변경
if [ -z "$1" ]; then
    echo "Check Parking Number: bash simulate.sh <parking_number>"
    exit 1
fi
PARKING_NUMBER=$1
export MAP_1_PARKING_AREA=$PARKING_NUMBER

# 시뮬 실행
source ~/catkin_ws/devel/setup.bash
roslaunch racecar_simulator simulate.launch map_number:=1
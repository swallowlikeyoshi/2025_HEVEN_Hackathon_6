cmake_minimum_required(VERSION 3.0.2)
project(formula)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  fs_msgs
  nav_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy roscpp fs_msgs nav_msgs geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

cmake_minimum_required(VERSION 3.0.2)
project(rrt_informed)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs 
  geometry_msgs
  
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES hybrid_Astar
#  CATKIN_DEPENDS rospy std_msgs
#  DEPENDS system_lib
)


include_directories(
# include
  launch
  ${catkin_INCLUDE_DIRS}
)


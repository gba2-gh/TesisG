cmake_minimum_required(VERSION 2.8.3)
project(blur_image)

## Add support for C++11, supported in ROS Kinetic and newer
# add_definitions(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

SET(CUDA_NVCC_FLAGS "-arch=sm_30" CACHE STRING "nvcc flags" FORCE) ##sm;gpu arch
SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
SET(LIB_TYPE STATIC) 
CUDA_ADD_LIBRARY(libBLUR ${LIB_TYPE} src/student_func.cu)  ##cuda-c source file
 

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES blur_image
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(blur_image_node src/main.cpp)
add_dependencies(blur_image_node libBLUR)


TARGET_LINK_LIBRARIES(blur_image_node
${catkin_LIBRARIES}
${PCL_LIBRARIES}
${OpenCV_LIBS}
libBLUR

)



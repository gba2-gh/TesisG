cmake_minimum_required(VERSION 2.8.3)
project(vhgw_test)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  )

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

SET(CUDA_NVCC_FLAGS "-arch=sm_30" CACHE STRING "nvcc flags" FORCE) ##sm;gpu arch
SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
SET(LIB_TYPE STATIC) 
CUDA_ADD_LIBRARY(color_segmenter_lib STATIC src/color_segmenter.cu)  ##cuda-c source file

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(vhgw_node src/vhgw_node.cpp)
add_dependencies(vhgw_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} color_segmenter_lib)

target_link_libraries(vhgw_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  color_segmenter_lib
)

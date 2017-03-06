cmake_minimum_required(VERSION 2.8.11)
project(CALIB_CAM)

SET(VERSION_REQ_OCV "3.1")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV ${VERSION_REQ_OCV} REQUIRED)
link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(calibrate calib_intrinsic.cpp popt_pp.h)
target_link_libraries(calibrate ${OpenCV_LIBS} "-lpopt")

add_executable(calibrate_stereo calib_stereo.cpp)
target_link_libraries(calibrate_stereo ${OpenCV_LIBS} "-lpopt")

add_executable(record_depth record_depth.cpp)
target_link_libraries(record_depth ${OpenCV_LIBS} "-lpopt")

add_executable(save_depth_img depth_img.cpp)
target_link_libraries(save_depth_img ${OpenCV_LIBS} "-lpopt")

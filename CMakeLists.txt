cmake_minimum_required(VERSION 3.25)

project(fisheye2cylindrical)
set (CMAKE_BUILD_TYPE "Debug") 
set(OPENCV_DIRS "D:/tools/opencv_3.0.0/build")
set(EIGEN_DIRS  "D:/tools/eigen3")

aux_source_directory(. SRC_LIST)

SET(INCLUDE_DIR .
               ${OPENCV_DIRS}/include
               ${OPENCV_DIRS}/include/opencv
               ${OPENCV_DIRS}/include/opencv2
               ${EIGEN_DIRS}/Eigen)

include_directories(${INCLUDE_DIR})

set(OPENCV_LIBS D:/tools/opencv_3.0.0/build_mingw/lib)
add_executable(fisheye2cylindrical ${SRC_LIST})

target_link_libraries(fisheye2cylindrical
        ${OPENCV_LIBS}/libopencv_ts300.a
        ${OPENCV_LIBS}/libopencv_world300.dll.a)

cmake_minimum_required(VERSION 3.16)
project(pc_viewer)

# build in release mode by default
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)

# Find ZED SDK
find_package(ZED 3 REQUIRED)
find_package(OpenCV REQUIRED)
# Find Open3D
find_package(Open3D REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)





add_library(yolov8 src/yolov8/yolov8_seg.cpp src/yolov8/yolov8_utils.cpp)
target_include_directories(yolov8 PUBLIC include/yolov8 ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolov8 ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY}
                      ${OpenCV_LIBRARIES} ${Open3D_LIBRARIES})


add_executable(pc_viewer src/viewer.cpp)
target_include_directories(pc_viewer PUBLIC include ${ZED_INCLUDE_DIRS} ${Open3D_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pc_viewer ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY}
                      ${Open3D_LIBRARIES} ${OpenCV_LIBRARIES} yolov8)

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

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

add_executable(pc_viewer src/viewer.cpp)
add_executable(refine_calibration src/refine_calibration.cpp)

target_link_libraries(pc_viewer ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY}
                      ${Open3D_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(refine_calibration ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY}
                      ${Open3D_LIBRARIES} ${OpenCV_LIBRARIES})
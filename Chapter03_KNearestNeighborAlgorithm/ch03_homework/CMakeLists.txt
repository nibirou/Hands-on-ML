cmake_minimum_required(VERSION 3.15)
project(KNN_MNIST)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(knn_mnist main.cpp)

# 【新增】强制 VSCode 运行程序时，工作目录为项目根目录，确保能找到相对路径的数据文件
set(CMAKE_VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

target_link_libraries(knn_mnist PRIVATE Eigen3::Eigen ${OpenCV_LIBS})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(knn_mnist PRIVATE OpenMP::OpenMP_CXX)
endif()
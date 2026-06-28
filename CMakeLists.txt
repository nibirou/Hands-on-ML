cmake_minimum_required(VERSION 3.15)
project(KNN_MNIST)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 Eigen3 和 OpenCV
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(knn_mnist main.cpp)

# 链接库
target_link_libraries(knn_mnist PRIVATE Eigen3::Eigen ${OpenCV_LIBS})

# 开启 OpenMP 多线程加速 (MinGW 的 GCC 完美支持 OpenMP)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(knn_mnist PRIVATE OpenMP::OpenMP_CXX)
endif()
#include <iostream>
#include <Eigen/Dense>

int main() {
    // ==========================================
    // 选项A 两个对角矩阵之间相乘一定可交换
    // ==========================================
    std::cout << "--- 选项A ---" << std::endl;
    
    // 定义对角线元素
    Eigen::Vector3d d1_diag(1, 2, 3);
    Eigen::Vector3d d2_diag(4, 5, 6);
    
    // 构造对角矩阵
    Eigen::Matrix3d D1 = d1_diag.asDiagonal();
    Eigen::Matrix3d D2 = d2_diag.asDiagonal();
    
    // 计算乘积
    Eigen::Matrix3d result1 = D1 * D2;
    Eigen::Matrix3d result2 = D2 * D1;
    
    // 输出结果
    std::cout << "D1 * D2:\n" << result1 << std::endl;
    std::cout << "D2 * D1:\n" << result2 << std::endl;
    
    // 检查两个乘积是否相等 (因为都是整数，可以直接用 ==)
    bool is_equal_A = (result1 == result2);
    std::cout << "Are D1 * D2 and D2 * D1 equal? " << (is_equal_A ? "True" : "False") << std::endl;


    // ==========================================
    // 选项B 矩阵与向量的乘法满足分配律
    // ==========================================
    std::cout << "\n--- 选项B ---" << std::endl;
    
    // 定义矩阵A
    Eigen::Matrix2d A;
    A << 1, 2,
         3, 4;
         
    // 定义向量x, y
    Eigen::Vector2d x(2, 2);
    Eigen::Vector2d y(3, 3);
    
    // 分别计算分配律两边
    Eigen::Vector2d result3 = A * (x + y);
    Eigen::Vector2d result4 = A * x + A * y;
    
    std::cout << "分配律左边:\n" << result3 << std::endl;
    std::cout << "分配律右边:\n" << result4 << std::endl;
    
    // 检查两边是否相等
    bool is_equal_B = (result3 == result4);
    std::cout << "Are they equal? " << (is_equal_B ? "True" : "False") << std::endl;


    // ==========================================
    // 选项C 矩阵对向量的点乘满足结合律 错误 因为点积内积可能是标量
    // ==========================================
    std::cout << "\n--- 选项C ---" << std::endl;
    
    // 重新定义向量 x, y (避免与选项B的变量冲突)
    Eigen::Vector2d x_c(1, 0);
    Eigen::Vector2d y_c(0, 1);
    
    // 计算 x.dot(y)
    double dot_xy = x_c.dot(y_c);
    
    // product1: (A * x) 点乘 y，结果是标量 (double)
    double product1 = (A * x_c).dot(y_c); 
    
    // product2: A 乘以 (x 点乘 y)，结果是矩阵 (Matrix2d)
    Eigen::Matrix2d product2 = A * dot_xy; 
    
    std::cout << "x.dot(y): " << dot_xy << std::endl;
    std::cout << "Product 1 (Scalar): " << product1 << std::endl;
    std::cout << "Product 2 (Matrix):\n" << product2 << std::endl;
    
    // 在C++中，由于强类型限制，标量(double)和矩阵(Matrix2d)无法直接比较。
    // 这里直接输出 False，并解释原因。
    std::cout << "Are they equal? False (Type mismatch: Scalar vs Matrix)" << std::endl;

    return 0;
}

// 编译命令
// g++ -I /usr/include/eigen3 ch02_homework2_eigen.cpp -o ch02_homework2_eigen.out
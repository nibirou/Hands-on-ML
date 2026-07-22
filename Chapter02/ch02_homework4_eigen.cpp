#include <iostream>
#include <Eigen/Dense>
#include <random>

int main() {
    // 设置随机数生成器 (对应 numpy.random.rand)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    Eigen::Matrix3d A;
    
    // 生成一个可逆的随机矩阵
    do {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                A(i, j) = dis(gen);
    } while (std::abs(A.determinant()) < 1e-9); // 确保矩阵行列式非0，矩阵可逆

    // 计算转置和逆
    Eigen::Matrix3d A_T = A.transpose();
    Eigen::Matrix3d inv_A = A.inverse();
    
    Eigen::Matrix3d inv_A_T = inv_A.transpose();  // (A^{-1})^T
    Eigen::Matrix3d inv_A_T_2 = A_T.inverse();    // (A^T)^{-1}

    // 验证两者是否相等 (Eigen 的 isApprox 会自动处理浮点数精度误差)
    bool is_equal = inv_A_T.isApprox(inv_A_T_2);
    std::cout << "原矩阵的转置矩阵的逆矩阵 与 原矩阵的逆矩阵的转置矩阵 是否相等: " 
              << (is_equal ? "True" : "False") << std::endl;

    return 0;
}

// 编译代码
// g++ -I /usr/include/eigen3 ch02_homework4_eigen.cpp -o ch02_homework4_eigen.out
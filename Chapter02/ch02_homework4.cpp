#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <iomanip>

// 定义 N x N 矩阵类型
using Matrix = std::vector<std::vector<double>>;

// ==========================================
// 基础矩阵运算函数
// ==========================================

// 1. 矩阵转置
Matrix transpose(const Matrix& m) {
    int n = m.size();
    Matrix res(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i][j] = m[j][i];
        }
    }
    return res;
}

// 2. 高斯-约当消元法求逆，并同时计算行列式
// 如果矩阵不可逆，将抛出异常
Matrix inverse(const Matrix& A, double& det) {
    int n = A.size();
    
    // 构造增广矩阵 [A | I]，大小为 n x 2n
    Matrix aug(n, std::vector<double>(2 * n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug[i][j] = A[i][j];
        }
        aug[i][n + i] = 1.0; // 右侧为单位矩阵 I
    }
    
    det = 1.0;
    const double EPS = 1e-9; // 浮点数判零阈值
    
    for (int i = 0; i < n; ++i) {
        // 步骤 A: 部分选主元 (Partial Pivoting) 以提高数值稳定性
        int max_row = i;
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(aug[k][i]) > std::abs(aug[max_row][i])) {
                max_row = k;
            }
        }
        
        // 如果主元接近 0，说明矩阵是奇异的（不可逆）
        if (std::abs(aug[max_row][i]) < EPS) {
            throw std::runtime_error("Matrix is singular (determinant is 0), cannot be inverted.");
        }
        
        // 交换当前行和主元所在行
        if (max_row != i) {
            std::swap(aug[i], aug[max_row]);
            det = -det; // 行交换会改变行列式的符号
        }
        
        // 记录主元，用于计算行列式
        det *= aug[i][i];
        
        // 步骤 B: 将主元行归一化，使 aug[i][i] = 1
        double pivot = aug[i][i];
        for (int j = 0; j < 2 * n; ++j) {
            aug[i][j] /= pivot;
        }
        
        // 步骤 C: 消去其他行的第 i 列元素
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = aug[k][i];
                for (int j = 0; j < 2 * n; ++j) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }
    
    // 步骤 D: 提取增广矩阵的右半部分，即为逆矩阵 A^{-1}
    Matrix inv(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            inv[i][j] = aug[i][n + j];
        }
    }
    
    return inv;
}

// 3. 判断两个矩阵是否相等 (处理浮点数精度误差)
bool isMatEqual(const Matrix& A, const Matrix& B, double eps = 1e-7) {
    int n = A.size();
    if (n != (int)B.size()) return false;
    for (int i = 0; i < n; ++i) {
        if (n != (int)A[i].size() || n != (int)B[i].size()) return false;
        for (int j = 0; j < n; ++j) {
            if (std::abs(A[i][j] - B[i][j]) > eps) {
                return false;
            }
        }
    }
    return true;
}

// 辅助函数：打印矩阵 (仅用于调试或小矩阵展示)
void printMatrix(const std::string& name, const Matrix& m) {
    std::cout << name << ":\n";
    for (const auto& row : m) {
        for (double val : row) {
            std::cout << std::setw(10) << val << " ";
        }
        std::cout << "\n";
    }
}

// ==========================================
// 主函数
// ==========================================
int main() {
    // 设定矩阵维度 N (可以修改为任意正整数，例如 4, 5, 10)
    const int N = 4; 
    
    std::cout << "正在验证 N = " << N << " 的矩阵性质...\n";

    // 设置随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0); // 扩大范围以避免全是极小值

    Matrix A(N, std::vector<double>(N));
    double detA = 0.0;

    // 生成一个可逆的随机矩阵 (行列式绝对值大于阈值)
    int attempts = 0;
    do {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i][j] = dis(gen);
            }
        }
        // 尝试求逆以获取行列式，如果不可逆会抛出异常，被 catch 捕获后重试
        try {
            Matrix temp_inv = inverse(A, detA);
            break; // 成功求逆，跳出循环
        } catch (const std::exception&) {
            attempts++;
        }
    } while (std::abs(detA) < 1e-5 && attempts < 100);

    if (attempts >= 100) {
        std::cerr << "Failed to generate an invertible matrix after 100 attempts.\n";
        return 1;
    }

    std::cout << "成功生成可逆矩阵 A, 行列式 det(A) = " << detA << "\n\n";

    // 1. 计算 A 的转置: A^T
    Matrix A_T = transpose(A);
    
    // 2. 计算 A 的逆: A^{-1}
    double detA_inv_temp; // 这里不需要 det，但函数签名需要
    Matrix inv_A = inverse(A, detA_inv_temp);
    
    // 3. 计算 (A^{-1})^T
    Matrix inv_A_T = transpose(inv_A);
    
    // 4. 计算 (A^T)^{-1}
    double detA_T;
    Matrix inv_A_T_2 = inverse(A_T, detA_T);

    // 验证两者是否相等
    bool is_equal = isMatEqual(inv_A_T, inv_A_T_2);
    
    std::cout << "==================================================\n";
    std::cout << "验证结果:\n";
    std::cout << "原矩阵的转置矩阵的逆矩阵 (A^T)^{-1} \n";
    std::cout << "与 \n";
    std::cout << "原矩阵的逆矩阵的转置矩阵 (A^{-1})^T \n";
    std::cout << "是否相等: " << (is_equal ? "True (相等)" : "False (不相等)") << "\n";
    std::cout << "==================================================\n";

    // 如果需要查看具体矩阵，可以取消下面两行的注释 (建议 N <= 5 时查看)
    // printMatrix("(A^{-1})^T", inv_A_T);
    // printMatrix("(A^T)^{-1}", inv_A_T_2);

    return 0;
}

// 编译命令
// g++ -std=c++11 ch02_homework4.cpp -o ch02_homework4.out
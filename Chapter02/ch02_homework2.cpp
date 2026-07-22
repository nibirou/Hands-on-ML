#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

void printMatrix(const std::string& name, const Matrix& mat) {
    std::cout << name << ":\n";
    for (const auto& row : mat) {
        for (double val : row) {
            std::cout << std::setw(8) << val << " ";
        }
        std::cout << "\n";
    }
}

void printVector(const std::string& name, const Vector& vec) {
    std::cout << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << "]\n";
}

Matrix matMul(const Matrix& A, const Matrix& B) {
    int rowsA = A.size(), colsA = A[0].size(), colsB = B[0].size();
    Matrix C(rowsA, std::vector<double>(colsB, 0.0));
    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsB; ++j)
            for (int k = 0; k < colsA; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

Vector matVecMul(const Matrix& A, const Vector& v) {
    int rows = A.size(), cols = A[0].size();
    Vector res(rows, 0.0);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            res[i] += A[i][j] * v[j];
    return res;
}

double vecDot(const Vector& a, const Vector& b) {
    double res = 0.0;
    for (size_t i = 0; i < a.size(); ++i) res += a[i] * b[i];
    return res;
}

Vector vecAdd(const Vector& a, const Vector& b) {
    Vector res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] + b[i];
    return res;
}

Matrix matScale(double scalar, const Matrix& A) {
    Matrix res = A;
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[0].size(); ++j)
            res[i][j] *= scalar;
    return res;
}

bool isMatEqual(const Matrix& A, const Matrix& B, double eps = 1e-9) {
    if (A.size() != B.size() || A[0].size() != B[0].size()) return false;
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[0].size(); ++j)
            if (std::abs(A[i][j] - B[i][j]) > eps) return false;
    return true;
}

bool isVecEqual(const Vector& a, const Vector& b, double eps = 1e-9) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::abs(a[i] - b[i]) > eps) return false;
    return true;
}

int main() {
    // ==========================================
    // 选项A 两个对角矩阵之间相乘一定可交换
    // ==========================================
    std::cout << "--- 选项A ---" << std::endl;
    
    Matrix D1 = {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};
    Matrix D2 = {{4, 0, 0}, {0, 5, 0}, {0, 0, 6}};
    
    Matrix resA1 = matMul(D1, D2);
    Matrix resA2 = matMul(D2, D1);
    
    printMatrix("D1 * D2", resA1);
    printMatrix("D2 * D1", resA2);
    // 注意：下面这行必须完整复制，包含末尾的 << "\n\n";
    std::cout << "Are D1 * D2 and D2 * D1 equal? " << (isMatEqual(resA1, resA2) ? "True" : "False") << "\n\n";

    // ==========================================
    // 选项B 矩阵与向量的乘法满足分配律
    // ==========================================
    std::cout << "--- 选项B ---" << std::endl;
    
    Matrix A = {{1, 2}, {3, 4}};
    Vector x = {2, 2};
    Vector y = {3, 3};
    
    Vector resB_left = matVecMul(A, vecAdd(x, y));
    Vector resB_right = vecAdd(matVecMul(A, x), matVecMul(A, y));
    
    printVector("分配律左边 A*(x+y)", resB_left);
    printVector("分配律右边 A*x + A*y", resB_right);
    std::cout << "Are they equal? " << (isVecEqual(resB_left, resB_right) ? "True" : "False") << "\n\n";

    // ==========================================
    // 选项C 矩阵对向量的点乘满足结合律 (错误)
    // ==========================================
    std::cout << "--- 选项C ---" << std::endl;
    
    Vector x_c = {1, 0};
    Vector y_c = {0, 1};
    
    double dot_xy = vecDot(x_c, y_c);
    double product1 = vecDot(matVecMul(A, x_c), y_c); 
    Matrix product2 = matScale(dot_xy, A); 
    
    std::cout << "x.dot(y) (Scalar): " << dot_xy << "\n";
    std::cout << "Product 1 (A*x)·y (Scalar): " << product1 << "\n";
    printMatrix("Product 2 A*(x·y) (Matrix)", product2);
    std::cout << "Are they equal? False (Type & Dimension mismatch: Scalar vs Matrix)\n";

    return 0;
}
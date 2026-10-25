#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <Eigen/Dense>

// 定义常用类型别名
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
using VectorXi = Eigen::VectorXi;

// ================= 辅助函数 =================

// 1. 加载CSV数据
void load_data(const std::string& filename, MatrixXd& X, VectorXi& y, int& K) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<std::vector<double>> raw_data;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::vector<double> row;
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        raw_data.push_back(row);
    }

    int N = raw_data.size();
    X.resize(N, 2);
    y.resize(N);
    int max_y = 0;

    for (int i = 0; i < N; ++i) {
        X(i, 0) = raw_data[i][0];
        X(i, 1) = raw_data[i][1];
        y(i) = static_cast<int>(raw_data[i][2]);
        if (y(i) > max_y) max_y = y(i);
    }
    K = max_y + 1; // 假设标签从0开始连续
}

// 2. 模型预测 (用于评估和网格生成)
VectorXi predict(const MatrixXd& X, const MatrixXd& W, const VectorXd& b, int K) {
    int N = X.rows();
    MatrixXd Z = X * W;
    Z.rowwise() += b.transpose(); // 广播加法

    MatrixXd Z_full(N, K);
    Z_full.leftCols(K - 1) = Z;
    Z_full.col(K - 1).setZero(); // 参考类得分为0

    // Softmax
    VectorXd Z_max = Z_full.rowwise().maxCoeff();
    MatrixXd exp_Z = (Z_full.colwise() - Z_max).array().exp();
    MatrixXd probs = exp_Z.array().colwise() / exp_Z.rowwise().sum().array();

    // Argmax
    VectorXi y_pred(N);
    for (int i = 0; i < N; ++i) {
        probs.row(i).maxCoeff(&y_pred(i));
    }
    return y_pred;
}

// ================= 主程序 =================

int main() {
    // 路径配置 (请根据实际情况修改)
    std::string data_path = "G:/pythonProject/Hands-on-ML/Chapter06/ch06_homework/lr_dataset.csv";
    std::string grid_out_path = "G:/pythonProject/Hands-on-ML/Chapter06/ch06_homework/grid_data.csv";
    std::string scatter_out_path = "G:/pythonProject/Hands-on-ML/Chapter06/ch06_homework/scatter_data.csv";

    // 1. 数据加载
    MatrixXd X;
    VectorXi y;
    int K;
    load_data(data_path, X, y, K);
    int N = X.rows();
    int D = X.cols();

    std::cout << "Loaded " << N << " samples, " << D << " features, " << K << " classes." << std::endl;

    // 2. 参数初始化
    std::default_random_engine generator(42); // 固定随机种子
    std::normal_distribution<double> distribution(0.0, 1.0);

    MatrixXd W(D, K - 1);
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < K - 1; ++j)
            W(i, j) = distribution(generator);
            
    VectorXd b = VectorXd::Zero(K - 1);

    // 3. 超参数
    double learning_rate = 0.1;
    int max_iter = 1000;
    double lambda_reg = 0.01;

    // 构建 One-Hot 编码矩阵 Y
    MatrixXd Y_onehot = MatrixXd::Zero(N, K);
    for (int i = 0; i < N; ++i) {
        Y_onehot(i, y(i)) = 1.0;
    }

    // 4. 训练循环
    for (int epoch = 0; epoch < max_iter; ++epoch) {
        // 前向传播
        MatrixXd Z = X * W;
        Z.rowwise() += b.transpose();

        MatrixXd Z_full(N, K);
        Z_full.leftCols(K - 1) = Z;
        Z_full.col(K - 1).setZero();

        VectorXd Z_max = Z_full.rowwise().maxCoeff();
        MatrixXd exp_Z = (Z_full.colwise() - Z_max).array().exp();
        MatrixXd probs = exp_Z.array().colwise() / exp_Z.rowwise().sum().array();

        
    }
}
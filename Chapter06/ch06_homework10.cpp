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
}
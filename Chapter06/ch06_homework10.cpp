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
    
}
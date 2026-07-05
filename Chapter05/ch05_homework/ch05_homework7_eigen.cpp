#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// 给输入矩阵添加一列全1，用于计算截距 (intercept)
MatrixXd add_intercept(const MatrixXd& X) {
    MatrixXd X_b(X.rows(), X.cols() + 1);
    X_b << MatrixXd::Ones(X.rows(), 1), X;
    return X_b;
}

// 训练线性回归模型 (使用 QR 分解求解最小二乘法，数值比直接求逆矩阵更稳定)
VectorXd train_linreg(const MatrixXd& X, const VectorXd& y) {
    MatrixXd X_b = add_intercept(X);
    return X_b.colPivHouseHolderQr().solve(y);
}

// 计算均方误差 (MSE)
double calc_mse(const MatrixXd& X, const VectorXd& y, const VectorXd& w) {
    MatrixXd X_b = add_intercept(X);
    VectorXd y_pred = X_b * w;
    return (y - y_pred).array().square().mean();
}

int main() {
    // 1. 从源文件加载数据
    ifstream file("D:/Download/USA_Housing.csv");
    if (!file.is_open()) {
        cerr << "无法打开文件，请检查路径！" << endl;
        return -1;
    }

    string line;
    vector<string> header;
    vector<vector<double>> data_vec;

    // 读取表头
    if (getline(file, line)) {
        stringstream ss(line);
        string val;
        while (getline(ss, val, ',')) {
            // 清理可能的 Windows 回车符 \r
            if (!val.empty() && val.back() == '\r') val.pop_back();
            header.push_back(val);
        }
    }

    // 读取数据
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string val;
        vector<double> row;
        while (getline(ss, val, ',')) {
            if (!val.empty() && val.back() == '\r') val.pop_back();
            try {
                row.push_back(stod(val));
            } catch (...) {
                cerr << "数据转换错误，请确保 CSV 仅包含数字。" << endl;
                return -1;
            }
        }
        data_vec.push_back(row);
    }
    file.close();

    if (data_vec.empty()) {
        cerr << "数据为空！" << endl;
        return -1;
    }

    int rows = data_vec.size();
    int cols = data_vec[0].size();

    // 打印数据特征和标签
    cout << "数据特征：";
    for (int i = 0; i < cols - 1; ++i) {
        cout << header[i] << (i == cols - 2 ? "" : ", ");
    }
    cout << "\n数据标签：" << header.back() << "\n";
    cout << "数据总条数：" << rows << "\n\n";

    // 2. 转换为 Eigen 矩阵
    MatrixXd data(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data(i, j) = data_vec[i][j];
        }
    }

    // 3. 数据归一化 (StandardScaler)
    RowVectorXd mean = data.colwise().mean();
    // 计算标准差 (sklearn 默认使用 ddof=0，即除以 N)
    RowVectorXd var = (data.rowwise() - mean).array().square().colwise().mean();
    RowVectorXd stddev = var.sqrt();
    
    // 防止除以 0
    for (int i = 0; i < stddev.size(); ++i) {
        if (stddev(i) == 0) stddev(i) = 1.0;
    }
    MatrixXd data_scaled = (data.rowwise() - mean).array().rowwise() / stddev;

    // 划分输入和标签
    MatrixXd X = data_scaled.leftCols(cols - 1);
    VectorXd y = data_scaled.col(cols - 1);

    // 4. 使用交叉验证来选择最佳模型 (K-Fold, K=5)
    int k = 5;
    vector<pair<int, int>> folds;
    int current = 0;
    for (int i = 0; i < k; ++i) {
        int size = rows / k + (i < rows % k ? 1 : 0); // 处理不能整除的情况
        folds.push_back({current, current + size});
        current += size;
    }

    vector<double> mse_scores;
    cout << "各模型的均方误差：";
    for (int i = 0; i < k; ++i) {
        int test_start = folds[i].first;
        int test_end = folds[i].second;
        
        // 构建训练集和测试集索引
        vector<int> train_indices;
        for (int j = 0; j < k; ++j) {
            if (j != i) {
                for (int idx = folds[j].first; idx < folds[j].second; ++idx) {
                    train_indices.push_back(idx);
                }
            }
        }

        MatrixXd X_train_cv(train_indices.size(), cols - 1);
        VectorXd y_train_cv(train_indices.size());
        for (int idx = 0; idx < train_indices.size(); ++idx) {
            X_train_cv.row(idx) = X.row(train_indices[idx]);
            y_train_cv(idx) = y(train_indices[idx]);
        }

        MatrixXd X_test_cv(test_end - test_start, cols - 1);
        VectorXd y_test_cv(test_end - test_start);
        for (int idx = 0; idx < test_end - test_start; ++idx) {
            X_test_cv.row(idx) = X.row(test_start + idx);
            y_test_cv(idx) = y(test_start + idx);
        }

        // 训练并评估
        VectorXd w = train_linreg(X_train_cv, y_train_cv);
        double mse = calc_mse(X_test_cv, y_test_cv, w);
        mse_scores.push_back(mse);
        cout << mse << (i == k - 1 ? "" : ", ");
    }
    cout << "\n";

    // 找到最小的 MSE (对应 Python 中的负 MSE 取 max)
    double best_cv_mse = *min_element(mse_scores.begin(), mse_scores.end());
    cout << "最佳模型的均方误差：" << best_cv_mse << "\n\n";

    // 5. 在测试集上评估最佳模型 (80% 训练, 20% 测试)
    int train_size = rows * 4 / 5;
    int test_size = rows - train_size;

    MatrixXd X_train = X.topRows(train_size);
    VectorXd y_train = y.head(train_size);
    MatrixXd X_test = X.bottomRows(test_size);
    VectorXd y_test = y.tail(test_size);

    VectorXd best_model_w = train_linreg(X_train, y_train);
    double mse_test = calc_mse(X_test, y_test, best_model_w);

    cout << "最佳模型在测试集上的均方误差：" << mse_test << "\n\n";

    // 6. 输出最佳模型的参数
    cout << "最佳模型的参数：" << "\n";
    cout << "系数（斜率）：\n" << best_model_w.tail(cols - 1).transpose() << "\n";
    cout << "截距：" << best_model_w(0) << "\n";

    return 0;
}
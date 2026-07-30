#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <iterator>
#include <cctype>

using namespace std;

// ================= 1. 数据读取模块 =================

vector<string> read_header(ifstream& file) {
    string line;
    getline(file, line);
    vector<string> header;
    stringstream ss(line);
    string item;
    while (getline(ss, item, ',')) {
        item.erase(remove_if(item.begin(), item.end(), ::isspace), item.end());
        header.push_back(item);
    }
    return header;
}

vector<vector<double>> read_data(ifstream& file) {
    vector<vector<double>> data;
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        vector<double> row;
        stringstream ss(line);
        string item;
        while (getline(ss, item, ',')) {
            item.erase(remove_if(item.begin(), item.end(), ::isspace), item.end());
            if (!item.empty()) {
                row.push_back(stod(item));
            }
        }
        if (!row.empty()) data.push_back(row);
    }
    return data;
}

// ================= 2. 标准归一化 (StandardScaler) =================

struct StandardScaler {
    vector<double> mean;
    vector<double> scale;
    int n_features;

    void fit(const vector<vector<double>>& data) {
        n_features = data[0].size();
        mean.assign(n_features, 0.0);
        scale.assign(n_features, 0.0);
        int n_samples = data.size();
        for (int j = 0; j < n_features; ++j) {
            double sum = 0.0;
            for (int i = 0; i < n_samples; ++i) sum += data[i][j];
            mean[j] = sum / n_samples;
        }
        for (int j = 0; j < n_features; ++j) {
            double sum_sq = 0.0;
            for (int i = 0; i < n_samples; ++i) {
                double diff = data[i][j] - mean[j];
                sum_sq += diff * diff;
            }
            // sklearn 使用的是总体标准差 (ddof=0)
            scale[j] = sqrt(sum_sq / n_samples);
            if (scale[j] == 0.0) scale[j] = 1.0; // 防止除零错误
        }
    }

    vector<vector<double>> transform(const vector<vector<double>>& data) {
        int n_samples = data.size();
        vector<vector<double>> res(n_samples, vector<double>(n_features));
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_features; ++j) {
                res[i][j] = (data[i][j] - mean[j]) / scale[j];
            }
        }
        return res;
    }
};

// ================= 3. 线性代数求解器 (高斯消元法) =================

vector<double> solve_linear_system(vector<vector<double>> A, vector<double> b) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        // 寻找主元 (部分主元法)
        double maxEl = std::abs(A[i][i]);
        int maxRow = i;
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(A[k][i]) > maxEl) {
                maxEl = std::abs(A[k][i]);
                maxRow = k;
            }
        }
        for (int k = i; k < n; ++k) swap(A[maxRow][k], A[i][k]);
        swap(b[maxRow], b[i]);

        if (std::abs(A[i][i]) < 1e-12) continue; // 奇异矩阵处理

        // 消元
        for (int k = i + 1; k < n; ++k) {
            double c = -A[k][i] / A[i][i];
            for (int j = i; j < n; ++j) {
                if (i == j) A[k][j] = 0;
                else A[k][j] += c * A[i][j];
            }
            b[k] += c * b[i];
        }
    }

    // 回代
    vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b[i] / A[i][i];
        for (int k = i - 1; k >= 0; --k) {
            b[k] -= A[k][i] * x[i];
        }
    }
    return x;
}

// ================= 4. 线性回归模型 =================

struct LinearRegression {
    vector<double> coef;
    double intercept;

    void fit(const vector<vector<double>>& X, const vector<double>& y) {
        int n_samples = X.size();
        int n_features = X[0].size();
        
        vector<vector<double>> A(n_features, vector<double>(n_features, 0.0));
        vector<double> b(n_features, 0.0);

        // 构造正规方程: (X^T * X) * beta = X^T * y
        for (int i = 0; i < n_features; ++i) {
            for (int j = 0; j < n_features; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n_samples; ++k) sum += X[k][i] * X[k][j];
                A[i][j] = sum;
            }
            double sum_b = 0.0;
            for (int k = 0; k < n_samples; ++k) sum_b += X[k][i] * y[k];
            b[i] = sum_b;
        }

        coef = solve_linear_system(A, b);

        // 按照 sklearn 的方式计算截距 (intercept = y_mean - X_mean * coef)
        double y_mean = 0.0;
        for(double v : y) y_mean += v;
        y_mean /= n_samples;

        vector<double> x_mean(n_features, 0.0);
        for(int i=0; i<n_samples; ++i) {
            for(int j=0; j<n_features; ++j) {
                x_mean[j] += X[i][j];
            }
        }
        for(int j=0; j<n_features; ++j) x_mean[j] /= n_samples;

        intercept = y_mean;
        for(int j=0; j<n_features; ++j) {
            intercept -= x_mean[j] * coef[j];
        }
    }

    vector<double> predict(const vector<vector<double>>& X) {
        int n_samples = X.size();
        vector<double> preds(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            double sum = intercept;
            for (size_t j = 0; j < X[0].size(); ++j) sum += X[i][j] * coef[j];
            preds[i] = sum;
        }
        return preds;
    }
};

// ================= 5. 交叉验证辅助函数 =================

vector<vector<int>> k_fold_indices(int n_samples, int n_splits) {
    vector<vector<int>> folds(n_splits);
    int fold_sizes = n_samples / n_splits;
    int remainder = n_samples % n_splits;
    
    int current = 0;
    for (int i = 0; i < n_splits; ++i) {
        int fold_size = fold_sizes + (i < remainder ? 1 : 0);
        for (int j = 0; j < fold_size; ++j) {
            folds[i].push_back(current++);
        }
    }
    return folds;
}

// ================= 6. 主函数 =================

int main() {
    string filename = "/workspace/Quant/Hands-on-ML/Chapter05/USA_Housing.csv";
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开文件，请检查路径。" << endl;
        return -1;
    }

    vector<string> header = read_header(file);
    vector<vector<double>> lines = read_data(file);
    file.close();

    cout << "数据特征：";
    for (size_t i = 0; i < header.size() - 1; ++i) {
        cout << header[i];
        if (i < header.size() - 2) cout << ", ";
    }
    cout << "\n数据标签：" << header.back() << "\n";
    cout << "数据总条数：" << lines.size() << "\n\n";

    // 数据归一化
    StandardScaler scaler;
    scaler.fit(lines);
    vector<vector<double>> lines_scaled = scaler.transform(lines);

    // 划分输入和标签
    int n_samples = lines_scaled.size();
    int n_features = lines_scaled[0].size() - 1;
    vector<vector<double>> X(n_samples, vector<double>(n_features));
    vector<double> y(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) X[i][j] = lines_scaled[i][j];
        y[i] = lines_scaled[i][n_features];
    }

    // 5折交叉验证
    int n_splits = 5;
    auto folds = k_fold_indices(n_samples, n_splits);
    vector<double> mse_scores(n_splits);

    for (int f = 0; f < n_splits; ++f) {
        vector<vector<double>> X_train, X_test;
        vector<double> y_train, y_test;
        
        for (int i = 0; i < n_samples; ++i) {
            bool in_test = false;
            for (int idx : folds[f]) {
                if (i == idx) { in_test = true; break; }
            }
            if (in_test) {
                X_test.push_back(X[i]);
                y_test.push_back(y[i]);
            } else {
                X_train.push_back(X[i]);
                y_train.push_back(y[i]);
            }
        }

        LinearRegression model;
        model.fit(X_train, y_train);
        vector<double> y_pred = model.predict(X_test);
        
        double mse = 0.0;
        for (size_t i = 0; i < y_test.size(); ++i) {
            double diff = y_test[i] - y_pred[i];
            mse += diff * diff;
        }
        mse /= y_test.size();
        mse_scores[f] = mse;
    }

    cout << "各模型的均方误差：[";
    for(int i=0; i<n_splits; ++i) {
        cout << mse_scores[i];
        if (i < n_splits - 1) cout << ", ";
    }
    cout << "]\n";

    // 【修复点】：选择均方误差最小的模型 (对应原代码逻辑注释中的最小MSE)
    auto min_it = min_element(mse_scores.begin(), mse_scores.end());
    double best_mse = *min_it;
    int best_model_index = distance(mse_scores.begin(), min_it);

    cout << "最佳模型的均方误差：" << best_mse << "\n\n";

    // 在测试集上评估最佳模型 (80% 训练, 20% 测试)
    int train_size = n_samples * 4 / 5;
    vector<vector<double>> X_train, X_test;
    vector<double> y_train, y_test;

    for(int i=0; i<n_samples; ++i) {
        if (i < train_size) {
            X_train.push_back(X[i]);
            y_train.push_back(y[i]);
        } else {
            X_test.push_back(X[i]);
            y_test.push_back(y[i]);
        }
    }

    LinearRegression best_model;
    best_model.fit(X_train, y_train);
    vector<double> y_pred = best_model.predict(X_test);

    double mse_test = 0.0;
    for(size_t i=0; i<y_test.size(); ++i) {
        double diff = y_test[i] - y_pred[i];
        mse_test += diff * diff;
    }
    mse_test /= y_test.size();

    cout << "最佳模型在测试集上的均方误差：" << mse_test << "\n\n";

    cout << "最佳模型的参数：\n";
    cout << "系数（斜率）：[";
    for(size_t i=0; i<best_model.coef.size(); ++i) {
        cout << best_model.coef[i];
        if (i < best_model.coef.size() - 1) cout << ", ";
    }
    cout << "]\n";
    cout << "截距：" << best_model.intercept << "\n";

    return 0;
}

// g++ -std=c++11 ch05_homework7.cpp -o ch05_homework7.out
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <iomanip>
#include <filesystem> // 引入 filesystem 解决 Windows 中文路径问题

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

// 辅助函数：读取特征数据到 Eigen 矩阵
// 参数改为 std::filesystem::path 以支持中文路径
Eigen::MatrixXi read_x(const std::filesystem::path& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }
    
    std::vector<std::vector<int>> temp_data;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<int> row;
        std::stringstream ss(line);
        int val;
        while (ss >> val) row.push_back(val);
        if (!row.empty()) temp_data.push_back(row);
    }

    int rows = temp_data.size();
    int cols = temp_data[0].size();
    Eigen::MatrixXi mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = temp_data[i][j];
        }
    }
    return mat;
}

// 辅助函数：读取标签数据到 Eigen 向量
Eigen::VectorXi read_y(const std::filesystem::path& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }
    std::vector<int> temp_data;
    int val;
    while (file >> val) temp_data.push_back(val);
    
    Eigen::VectorXi vec(temp_data.size());
    for (size_t i = 0; i < temp_data.size(); ++i) vec(i) = temp_data[i];
    return vec;
}

class KNN {
private:
    int k;
    int label_num;
    Eigen::MatrixXi x_train;
    Eigen::VectorXi y_train;

public:
    KNN(int k, int label_num) : k(k), label_num(label_num) {}

    void fit(const Eigen::MatrixXi& x, const Eigen::VectorXi& y) {
        x_train = x;
        y_train = y;
    }

    int get_label(const Eigen::RowVectorXi& x) {
        // 利用 Eigen 的广播机制，一次性计算测试样本与所有训练样本的曼哈顿距离
        Eigen::VectorXi dists = (x_train.rowwise() - x).cwiseAbs().rowwise().sum();

        std::vector<std::pair<int, int>> dist_idx(x_train.rows());
        for (int i = 0; i < x_train.rows(); ++i) {
            dist_idx[i] = {dists(i), i};
        }

        // 只排序前 K 个，性能极高
        std::partial_sort(dist_idx.begin(), dist_idx.begin() + k, dist_idx.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<int> label_statistic(label_num, 0);
        for (int i = 0; i < k; ++i) {
            int label = y_train(dist_idx[i].second);
            label_statistic[label]++;
        }

        return std::max_element(label_statistic.begin(), label_statistic.end()) - label_statistic.begin();
    }

    Eigen::VectorXi predict(const Eigen::MatrixXi& x_test) {
        Eigen::VectorXi predicted_labels(x_test.rows());
        
        // 开启 OpenMP 多线程并行预测
        #pragma omp parallel for
        for (int i = 0; i < x_test.rows(); ++i) {
            predicted_labels(i) = get_label(x_test.row(i));
            
            if (i % 1000 == 0) {
                #pragma omp critical
                std::cout << "\r预测进度: " << i << " / " << x_test.rows() << std::flush;
            }
        }
        std::cout << "\r预测进度: " << x_test.rows() << " / " << x_test.rows() << " (完成)" << std::endl;
        return predicted_labels;
    }
};

int main() {
    std::cout << "正在加载数据..." << std::endl;
    // 使用 std::filesystem::path 处理包含中文的路径
    std::filesystem::path x_path = "./第3章 k近邻算法/mnist_x";
    std::filesystem::path y_path = "./第3章 k近邻算法/mnist_y";

    Eigen::MatrixXi m_x = read_x(x_path);
    Eigen::VectorXi m_y = read_y(y_path);

    // ================= 数据集可视化 =================
    std::cout << "正在显示第一张图像 (按任意键关闭窗口)..." << std::endl;
    Eigen::RowVectorXi first_img_vec = m_x.row(0);
    
    cv::Mat img(28, 28, CV_8UC1);
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            img.at<uchar>(i, j) = first_img_vec(i * 28 + j);
        }
    }
    
    cv::imshow("MNIST Sample (Press any key to close)", img);
    cv::waitKey(0); 
    cv::destroyAllWindows();

    std::cout << "正在划分和打乱数据集..." << std::endl;
    double ratio = 0.8;
    int total_size = m_x.rows();
    int split = static_cast<int>(total_size * ratio);

    std::vector<int> idx(total_size);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(0); 
    std::shuffle(idx.begin(), idx.end(), rng);

    Eigen::MatrixXi shuffled_x(total_size, m_x.cols());
    Eigen::VectorXi shuffled_y(total_size);
    for (int i = 0; i < total_size; ++i) {
        shuffled_x.row(i) = m_x.row(idx[i]);
        shuffled_y(i) = m_y(idx[i]);
    }

    Eigen::MatrixXi x_train = shuffled_x.topRows(split);
    Eigen::MatrixXi x_test = shuffled_x.bottomRows(total_size - split);
    Eigen::VectorXi y_train = shuffled_y.head(split);
    Eigen::VectorXi y_test = shuffled_y.tail(total_size - split);

    std::cout << "开始训练与测试..." << std::endl;
    for (int k_val = 1; k_val < 10; ++k_val) {
        KNN knn(k_val, 10);
        knn.fit(x_train, y_train);
        Eigen::VectorXi predicted_labels = knn.predict(x_test);
        
        int correct = (predicted_labels.array() == y_test.array()).count();
        double accuracy = static_cast<double>(correct) / y_test.size();
        
        std::cout << "K的取值为 " << k_val << ", 预测准确率为 " 
                  << std::fixed << std::setprecision(2) << (accuracy * 100) << "%" << std::endl;
    }

    return 0;
}
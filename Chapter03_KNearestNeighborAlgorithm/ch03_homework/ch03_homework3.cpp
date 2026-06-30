// # 习题3
// # KNN 算法中，我们采用了最常用的欧氏距离作为寻找邻居的标准。
// # 在哪些场景下，我们可能会用到其他距离度量，例如曼哈顿距离（Manhattan distance）

// # 把第 3 节实验中的距离改为曼哈顿距离，观察对分类效果的影响。
// # c++版本代码（不含外部库）
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <iomanip>

// 辅助函数：读取特征数据 (m_x)
std::vector<std::vector<int>> read_x(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }
    std::vector<std::vector<int>> data;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<int> row;
        std::stringstream ss(line);
        int val;
        while(ss >> val){
            row.push_back(val);
        }
        if(!row.empty()){
            data.push_back(row);
        }
    }
    return data;
}

// 辅助函数：读取标签数据 (m_y)
std::vector<int> read_y(const std::string& filename){
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }
    std::vector<int> data;
    int val;
    while (file >> val) {
        data.push_back(val);
    }
    return data;
}

class KNN {
    private:
    int k;
    int label_num;
    std::vector<std::vector<int>> x_train;
    std::vector<int> y_train;

    // 曼哈顿距离计算
    int distance(const std::vector<int>& a, const std::vector<int>& b) {
        int dist = 0;
        // 使用指针遍历可以稍微提升一点性能
        const int* ptr_a = a.data();
        const int* ptr_b = b.data();
        size_t size = a.size();
        for (size_t i = 0; i < size; ++i) {
            dist += std::abs(ptr_a[i] - ptr_b[i]);
        }
        return dist;
    }
    public:
        KNN(int k, int label_num) : k(k), label_num(label_num) {}

        void fit(const std::vector<std::vector<int>>& x, const std::vector<int>& y) {
            // 保存训练数据
            x_train = x;
            y_train = y;
        }

        std::vector<int> get_knn_indices(const std::vector<int>& x) {
            // 计算已知样本的距离，并保存 (距离, 索引) 对
            std::vector<std::pair<int, int>> dist_idx;
            dist_idx.reserve(x_train.size());
            for (size_t i = 0; i < x_train.size(); ++i) {
                dist_idx.emplace_back(distance(x_train[i], x), i);
            }
            
            // 优化：使用 partial_sort 只找出前 k 个最小的，比全排序 (argsort) 更快
            std::partial_sort(dist_idx.begin(), dist_idx.begin() + k, dist_idx.end(),
                [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                    return a.first < b.first;
                });

            // 提取前 K 个索引
            std::vector<int> knn_indices(k);
            for (int i = 0; i < k; ++i) {
                knn_indices[i] = dist_idx[i].second;
            }
            return knn_indices;
        }

        int get_label(const std::vector<int>& x) {
            std::vector<int> knn_indices = get_knn_indices(x);
            std::vector<int> label_statistic(label_num, 0);
            
            // 统计 K 个近邻的类别
            for (int index : knn_indices) {
                int label = y_train[index];
                label_statistic[label]++;
            }
            
            // 获取数量最多的类别 (相当于 np.argmax)
            return std::max_element(label_statistic.begin(), label_statistic.end()) - label_statistic.begin();
        }

        std::vector<int> predict(const std::vector<std::vector<int>>& x_test) {
            std::vector<int> predicted_test_labels(x_test.size());
            
            // 如果编译器支持 OpenMP，可以取消下面这行的注释以开启多线程加速
            // #pragma omp parallel for
            for (size_t i = 0; i < x_test.size(); ++i) {
                predicted_test_labels[i] = get_label(x_test[i]);
                
                // 打印进度，防止用户以为程序卡死
                if (i % 1000 == 0) {
                    std::cout << "\r预测进度: " << i << " / " << x_test.size() << std::flush;
                }
            }
            std::cout << "\r预测进度: " << x_test.size() << " / " << x_test.size() << " (完成)" << std::endl;
            return predicted_test_labels;
        }
    };

int main() {
    std::cout << "正在加载数据..." << std::endl;
    // 注意：请确保路径正确，C++中推荐使用正斜杠或双反斜杠
    std::vector<std::vector<int>> m_x = read_x("/workspace/Quant/Hands-on-ML/Chapter03_KNearestNeighborAlgorithm/mnist_x");
    std::vector<int> m_y = read_y("/workspace/Quant/Hands-on-ML/Chapter03_KNearestNeighborAlgorithm/mnist_y");

    /* 
    * 数据集可视化部分：
    * C++标准库没有 matplotlib。如需可视化，建议引入 OpenCV 库 (cv::imshow) 
    * 或使用 matplotlib-cpp 第三方库。此处为保持核心算法纯粹，予以省略。
    */

    std::cout << "正在划分和打乱数据集..." << std::endl;
    // 将数据集分为训练集和测试集
    double ratio = 0.8;
    int total_size = m_x.size();
    int split = static_cast<int>(total_size * ratio);

    // 打乱数据 (设置随机种子为0，与Python的 np.random.seed(0) 对应)
    std::vector<int> idx(total_size);
    std::iota(idx.begin(), idx.end(), 0);
    
    std::mt19937 rng(0); 
    std::shuffle(idx.begin(), idx.end(), rng);

    // 根据打乱后的索引重排数据
    std::vector<std::vector<int>> shuffled_x(total_size);
    std::vector<int> shuffled_y(total_size);
    for (int i = 0; i < total_size; ++i) {
        shuffled_x[i] = m_x[idx[i]];
        shuffled_y[i] = m_y[idx[i]];
    }

    // 划分训练集和测试集
    std::vector<std::vector<int>> x_train(shuffled_x.begin(), shuffled_x.begin() + split);
    std::vector<std::vector<int>> x_test(shuffled_x.begin() + split, shuffled_x.end());
    std::vector<int> y_train(shuffled_y.begin(), shuffled_y.begin() + split);
    std::vector<int> y_test(shuffled_y.begin() + split, shuffled_y.end());

    std::cout << "开始训练与测试..." << std::endl;
    // 遍历 K 值
    for (int k_val = 1; k_val < 10; ++k_val) {
        KNN knn(k_val, 10);
        knn.fit(x_train, y_train);
        std::vector<int> predicted_labels = knn.predict(x_test);
        
        // 计算准确率
        int correct = 0;
        for (size_t i = 0; i < predicted_labels.size(); ++i) {
            if (predicted_labels[i] == y_test[i]) {
                correct++;
            }
        }
        
        double accuracy = static_cast<double>(correct) / y_test.size();
        std::cout << "K的取值为 " << k_val << ", 预测准确率为 " 
                << std::fixed << std::setprecision(2) << (accuracy * 100) << "%" << std::endl;
    }

    return 0;
}

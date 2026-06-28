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
    
}
#include <iostream>
#include <cmath>
#include <iomanip>
#include <array>

// 定义二维向量类型，使用 std::array 比 std::vector 更高效
using Vec2 = std::array<double, 2>;

// 定义梯度函数 ∇f(x, y)
Vec2 gradient_f(double x, double y) {
    return {2.0 * x, 3.0 * y * y};
}

// 计算向量的模长 (L2 Norm)
double norm(const Vec2& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1]);
}

int main() {
    // 1. 测试点与梯度计算
    double x0 = 1.0, y0 = 1.0;
    Vec2 grad = gradient_f(x0, y0);
    
    // 梯度方向的单位向量
    double grad_norm = norm(grad);
    Vec2 grad_dir = {grad[0] / grad_norm, grad[1] / grad_norm};
    
    std::cout << "梯度方向单位向量: [" << grad_dir[0] << ", " << grad_dir[1] << "]\n\n";

    // 2. 生成不同方向并计算方向导数
    // 优化：不需要像 NumPy 那样在内存中生成 1000 个向量的数组，
    // 直接在循环中边生成边计算，空间复杂度从 O(N) 降为 O(1)。
    int num_directions = 1000;
    double max_deriv = -1e9; // 初始化为一个极小的值
    double max_theta = 0.0;
    
    // 使用 std::acos(-1.0) 获取 PI，保证在 Windows/Linux 等所有平台下跨平台兼容
    const double PI = std::acos(-1.0); 

    for (int i = 0; i < num_directions; ++i) {
        // 对应 np.linspace(0, 2*np.pi, 1000)
        double theta = 2.0 * PI * i / (num_directions - 1);
        double dx = std::cos(theta);
        double dy = std::sin(theta);
        
        // 计算方向导数 (梯度与方向单位向量的点积)
        double deriv = grad[0] * dx + grad[1] * dy;
        
        // 寻找最大值
        if (deriv > max_deriv) {
            max_deriv = deriv;
            max_theta = theta;
        }
    }

    // 3. 计算角度并输出
    // 计算梯度方向的角度 (弧度转角度)
    // 注意：使用 std::atan2(y, x) 比 std::atan(y/x) 更好，因为它能自动处理四个象限
    double grad_angle_rad = std::atan2(grad_dir[1], grad_dir[0]);
    double grad_angle_deg = grad_angle_rad * 180.0 / PI;
    
    double max_theta_deg = max_theta * 180.0 / PI;

    // 设置输出格式：保留两位小数
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "梯度方向：θ = " << grad_angle_deg << "°\n";
    std::cout << "方向导数最大值对应的角度：θ = " << max_theta_deg << "°\n";

    return 0;
}
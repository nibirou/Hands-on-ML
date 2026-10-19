import numpy as np
import matplotlib.pyplot as plt

# |  $n_1$  |  $n_2$  |  $n_3$  |  $n_4$  |  $p_1$  |  $p_2$  |  $p_3$  |  $p_4$  |
# | 0.15    | 0.21    | 0.74    | 0.45    | 0.71    | 0.48    | 0.52    | 0.34    |

# 8个样本：前4个是负例(n1~n4)，后4个是正例(p1~p4)
y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_pred = np.array([0.15, 0.21, 0.74, 0.45, 0.71, 0.48, 0.52, 0.34])

def calculate_roc_auc(y_true, y_pred):
    """
    计算ROC曲线和AUC值
    
    参数:
    y_true -- 真实标签（0/1）
    y_pred -- 预测为正类的概率
    
    返回:
    auc -- AUC值
    fpr -- 假阳性率数组
    tpr -- 真阳性率数组
    """
    # 1. 按预测概率从高到低排序
    sorted_indices = np.argsort(y_pred)[::-1]  # 降序排列索引
    y_true_sorted = y_true[sorted_indices]    # 排序后的真实标签
    
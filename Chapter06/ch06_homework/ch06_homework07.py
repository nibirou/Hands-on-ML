import numpy as np
import matplotlib.pyplot as plt

y_true = np.array([0, 0, 1, 1])  # 手动转换为0/1标签
y_pred = np.array([0.71, 0.48, 0.52, 0.34])  # 预测概率

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
    y_pred_sorted = y_pred[sorted_indices]    # 排序后的预测概率

    

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
    print("sorted_indices:", sorted_indices)
    y_true_sorted = y_true[sorted_indices]    # 排序后的真实标签
    print("y_true_sorted:", y_true_sorted)
    
    # y_pred_sorted = y_pred[sorted_indices]    # 排序后的预测概率
    # print("y_pred_sorted:", y_pred_sorted)
    
    # 2. 计算累积 TP 和 FP
    tp = np.cumsum(y_true_sorted)
    print("tp:", tp)
    fp = np.cumsum(1 - y_true_sorted)
    print("fp:", fp)
    
    # 3. 计算 TPR 和 FPR
    total_pos = tp[-1]   # 正例总数 = 4
    print("total_pos:", total_pos)
    total_neg = fp[-1]   # 负例总数 = 4
    print("total_neg:", total_neg)
    tpr = tp / total_pos
    print("tpr:", tpr)
    fpr = fp / total_neg
    print("fpr:", fpr)
    
    # 4. 添加起点 (0, 0)
    tpr = np.concatenate([[0], tpr])
    print("tpr:", tpr)
    fpr = np.concatenate([[0], fpr])
    print("fpr:", fpr)
    
    # 5. 梯形法计算 AUC
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * tpr[i]
        
    print("auc:", auc)
    return auc, fpr, tpr

roc_auc, fpr, tpr = calculate_roc_auc(y_true, y_pred)
print(f"AUC值为：{roc_auc:.4f}")

# 打印排序后的过程，方便理解
sorted_idx = np.argsort(y_pred)[::-1]
print("\n按概率降序排列：")
print(f"{'样本':<6}{'预测概率':<10}{'真实标签':<10}{'TP':<6}{'FP':<6}{'TPR':<8}{'FPR':<8}")
tp_cum, fp_cum = 0, 0
for i, idx in enumerate(sorted_idx):
    tp_cum += y_true[idx]
    fp_cum += (1 - y_true[idx])
    print(f"{idx+1:<6}{y_pred[idx]:<10}{y_true[idx]:<10}{tp_cum:<6}{fp_cum:<6}{tp_cum/4:<8}{fp_cum/4:<8}")

# 绘图
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random (AUC=0.5)')
plt.plot(fpr, tpr, color='darkorange', marker='o', label=f'ROC (AUC={roc_auc:.4f})')

# 标注每个阈值点
for i in range(len(fpr)):
    plt.annotate(f'({fpr[i]:.2f},{tpr[i]:.2f})', 
                 (fpr[i], tpr[i]), textcoords="offset points", 
                 xytext=(5, 5), fontsize=8)

plt.xlabel('FPR (假阳性率)', fontsize=12)
plt.ylabel('TPR (真阳性率)', fontsize=12)
plt.title('ROC 曲线', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
    
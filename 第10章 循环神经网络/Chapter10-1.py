# 对于具有序列特征的数据，如温度、文本等，他们具有明显的前后关联。
# 同时这些关联的数据在序列中出现的位置可能间隔非常远，比如文章在开头和结尾描写了同一个事物，如果用CNN来提取关联的话，
# 其卷积核的大小需要和序列的长度匹配。当数据序列较长时，会大大增加网络复杂度和训练难度
# 因此，引入循环神经网络RNN，充分利用数据的序列性质，从前到后分析数据、提取关联

# RNN因为在反向求导时，随着反向传播步数增加，由于求导链式法则，梯度中出现多层激活函数导数与权重项连乘
# 可能会出现梯度消失与梯度爆炸（实际上神经网络都这样）
# 梯度消失，模型收敛速度会变慢；梯度爆炸，模型梯度会迅速发散，参数变化幅度大，不收敛

# 为了防止梯度消失或爆炸，最简单的方法是裁剪梯度，为梯度设置上下限。当梯度过大或者过小时，采用上下限来替代梯度的值
# 还可以选用合适的激活函数并调整网络权重参数的初始值，使得两者乘积稳定在1附近。

# 因此，可以将网络中关联起相邻两步的激活函数和网络权重参数扩展成一个小的网络

# 门控循环单元(gated recurrent unit, GRU)

# 动手实现一个GRU模型，完成简单的时间序列预测任务

# 根据一段连续时间内采集的数据，分析其变化规律，预测数据走向

# <------------------------------------------------------------------------------>
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn

# 导入数据集
data = np.loadtxt('./第10章 循环神经网络/sindata_1000.csv', delimiter=',')
num_data = len(data)
split = int(0.8 * num_data)
print(f'数据集大小：{num_data}')
# 数据集可视化
plt.figure()
plt.scatter(np.arange(split), data[:split], color='blue', 
    s=10, label='training set')
plt.scatter(np.arange(split, num_data), data[split:], color='none', 
    edgecolor='orange', s=10, label='test set')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.savefig('./第10章 循环神经网络/training_test_split.png')
plt.show()

# 分割数据集
train_data = np.array(data[:split])
test_data = np.array(data[split:])

# <------------------------------------------------------------------------------>
# 在训练RNN模型时，虽然我们可以一把每个时间步数t单独输入，得到模型的预测值，但这样无法体现数据的序列相关性质
# 因此，通常会把一段时间序列整体作为输入，Pytorch中的GRU模块输出这段序列对应的中间变量。
# 如果要得到最后的输出，还需要将中间变量经过自定义的其他网络，这一点和CNN中卷积层负责提取特征、MLP负责根据特征
# 完成特定任务的做法非常相似。因此，在GRU之后拼接一个全连接层，通过中间变量序列来预测

# 输入序列长度
seq_len = 20
# 处理训练数据，把切分序列后多余的部分去掉
train_num = len(train_data) // (seq_len + 1) * (seq_len + 1)
train_data = np.array(train_data[:train_num]).reshape(-1, seq_len + 1, 1)
# print(train_data.shape)
np.random.seed(0)
torch.manual_seed(0)

x_train = train_data[:, :seq_len] # 形状为(num_data, seq_len, input_size)
y_train = train_data[:, 1: seq_len + 1]
print(f'训练序列数：{len(x_train)}')

# 转为PyTorch张量
x_train = torch.from_numpy(x_train).to(torch.float32)
y_train = torch.from_numpy(y_train).to(torch.float32)
x_test = torch.from_numpy(test_data[:-1]).to(torch.float32)
y_test = torch.from_numpy(test_data[1:]).to(torch.float32)

# 考虑到GRU的模型结构较为复杂，我们直接使用pytorch库中封装好的GRU模型
# 我们只 要为该模型提供两个参数，第一个参数input_size 表示输入 的维度 第二 参数hidden_size表示 中间变量 的维度，其余参数我们保持缺省值。

# 在前向传播时，GRU接收序列x和初始中间变量h。如果最开始我们不知道中间变量的值， RU 会自动将其初始
# 化为全零。 前向传播的输出 out 和 hidden，前者是 整个时间序列上中间变量的值 而后者
# 只包含最后一步 out[-1] 和hidden在GRU 内部的层数不同时会 有区别，但本节只使用
# 单层网络 因此不详细展开。感兴趣的读者可以参考 oyTorch 的官 文档 我们将 out作为最
# 后全连接层的输入， 得到预测值 再把预测值和 hidden 返回， hidden 将作为下次前向传
# 播的初始中间变量。

class GRU(nn.Module):
    # 包含PyTorch的GRU和拼接的MLP
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        # GRU模块
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size) 
        # 将中间变量映射到预测输出的MLP
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        # 前向传播
        # x的维度为(batch_size, seq_len, input_size)
        # GRU模块接受的输入为(seq_len, batch_size, input_size)
        # 因此需要对x进行变换
        # transpose函数可以交换x的坐标轴
        # out的维度是(seq_len, batch_size, hidden_size)
        out, hidden = self.gru(torch.transpose(x, 0, 1), hidden) 
        # 取序列最后的中间变量输入给全连接层
        out = self.linear(out.view(-1, hidden_size))
        return out, hidden
    
# 接下来，设置超参数并实例化GRU。在训练之前，我们还要强调时序模型在测试时与普通模型的区别。
# GRU在测试时，我们将输入的时间序列长度降为1，即只输入Xt,让GRU预测t+1时刻的值。
# 之后，不像普通的任务那样把所有测试数据都给模型
# 而是让GRU将自己预测的Xt+1作为输入，再预测t+2时刻的值，循环往复
# 这样的测试方式对模型在时序上的建模能力有相当高的要求， 否则就会很快因为预测值的误差累积，与真实值偏差很大。

# 超参数
input_size = 1 # 输入维度
output_size = 1 # 输出维度
hidden_size = 16 # 中间变量维度
learning_rate = 5e-4

# 初始化网络
gru = GRU(input_size, output_size, hidden_size)
gru_optim = torch.optim.Adam(gru.parameters(), lr=learning_rate)

# GRU测试函数，x和hidden分别是初始的输入和中间变量
def test_gru(gru, x, hidden, pred_steps):
    pred = []
    inp = x.view(-1, input_size)
    for i in range(pred_steps):
        gru_pred, hidden = gru(inp, hidden)
        pred.append(gru_pred.detach())
        inp = gru_pred
    return torch.concat(pred).reshape(-1)

# <------------------------------------------------------------------------------>
# 作为对比，使用相同的数据同步训练一个3层的MLP模型。该MLP模型同样将t到t+k时刻的xt,...,xt+k的数据拼接在一起作为输入
# 此时k被理解为输入的批量大小，并输出xt+1,...,xt+k+1的预测值，与GRU保持一致。

# MLP的超参数
hidden_1 = 32
hidden_2 = 16
mlp = nn.Sequential(
    nn.Linear(input_size, hidden_1),
    nn.ReLU(),
    nn.Linear(hidden_1, hidden_2),
    nn.ReLU(),
    nn.Linear(hidden_2, output_size)
)
mlp_optim = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

# MLP测试函数，相比于GRU少了中间变量
def test_mlp(mlp, x, pred_steps):
    pred = []
    inp = x.view(-1, input_size)
    for i in range(pred_steps):
        mlp_pred = mlp(inp)
        pred.append(mlp_pred.detach())
        inp = mlp_pred
    return torch.concat(pred).reshape(-1)

# 使用完全相同的数据训练GRU和MLP。由于已经有了序列长度，不再设置SGD的批量大小，直接将每个训练样本单独输入模型进行优化
max_epoch = 150
criterion = nn.functional.mse_loss
hidden = None # GRU的中间变量

# 训练损失
gru_losses = []
mlp_losses = []
gru_test_losses = []
mlp_test_losses = []

# 开始训练
with tqdm(range(max_epoch)) as pbar:
    for epoch in pbar:
        st = 0
        gru_loss = 0.0
        mlp_loss = 0.0
        # 随机梯度下降
        for X, y in zip(x_train, y_train):
            # 更新GRU模型
            # 我们不需要通过梯度回传更新中间变量
            # 因此将其从有梯度的部分分离出来
            if hidden is not None:
                hidden.detach_()
            gru_pred, hidden = gru(X[None, ...], hidden)
            gru_train_loss = criterion(gru_pred.view(y.shape), y)
            gru_optim.zero_grad()
            gru_train_loss.backward()
            gru_optim.step()
            gru_loss += gru_train_loss.item()
            # 更新MLP模型
            # 需要对输入的维度进行调整，变成(seq_len, input_size)的形式
            mlp_pred = mlp(X.view(-1, input_size))
            
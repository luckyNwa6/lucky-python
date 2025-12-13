# coding:utf8  # 指定 Python 文件编码为 UTF-8
import numpy as np  # 导入 numpy，用于生成随机数据
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
from matplotlib import pyplot as plt


# 使用numpy手动实现模拟一个线性层
# 实现一个自行构造的找规律任务
# 规律: x 是一个 5 维向量，如果第 1 个数 > 第 5 个数，则为正样本，否则为负样本

# 做个5分类任务 哪一维最大
class TorchModel(nn.Module):  # 定义一个继承自 nn.Module 的模型类

    def __init__(self, input_size):
        super(TorchModel, self).__init__()  # 调用父类构造函数

        self.linear = nn.Linear(input_size, 1)  # 定义一个线性层，将输入 5 维映射到 1 维

        self.activation = torch.sigmoid  # 使用 sigmoid 激活函数
        self.loss = nn.functional.mse_loss  # 使用 MSE 均方差作为损失函数

    def forward(self, x, y=None):
        # 定义前向传播过程
        x = self.linear(x)  # 输入通过线性层，shape: (batch_size, 5) -> (batch_size, 1)
        y_pred = self.activation(x)  # 对线性层输出做 sigmoid 激活
        if y is not None:  # 如果提供了标签 y，则计算损失
            return self.loss(y_pred, y)
        else:  # 否则返回预测值
            return y_pred


def build_sample():
    x = np.random.randn(5)  # 随机生成一个 5 维向量
    if x[0] > x[4]:  # 判断第 1 个数是否大于第 5 个数
        return x, 1  # 正样本
    else:
        return x, 0  # 负样本


def build_dataset(total_sample_num):
    X = []  # 存放所有样本的列表
    Y = []  # 存放所有标签的列表
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])  # 标签是标量，需要加 []
    # print(X)
    # print(Y)
    # 先用 np.array(X) 将列表转换为一个大的 NumPy 数组，然后 PyTorch 再将其转为 Tensor
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))


def evaluate(model):
    model.eval()  # 切换到评估模式
    test_sample_num = 100  # 测试样本数量
    x, y = build_dataset(test_sample_num)  # 构建测试集
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))  # 打印样本统计
    correct, wrong = 0, 0  # 初始化正确和错误计数
    with torch.no_grad():  # 禁用梯度计算
        y_pred = model(x)  # 得到模型预测
        for y_p, y_t in zip(y_pred, y):  # 遍历预测值和真实值
            if float(y_p) < 0.5 and int(y_t) == 0:  # 预测为 0 且真实为 0
                correct += 1
            elif float(y_p) >= 0.5 and int(y_t) == 1:  # 预测为 1 且真实为 1
                correct += 1
            else:
                wrong += 1  # 其他情况为错误
    print("正确预测个数：%d，正确率：%f" % (correct, correct / (correct + wrong)))  # 输出准确率
    return correct / (correct + wrong)  # 返回准确率


def main():
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每批次大小
    train_sample = 5000  # 训练样本数量
    input_size = 5  # 输入维度
    learning_rate = 0.001  # 学习率

    model = TorchModel(input_size)  # 创建模型实例

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义 Adam 优化器
    log = []  # 用于记录训练日志

    train_x, train_y = build_dataset(train_sample)  # 构建训练集

    for epoch in range(epoch_num):  # 训练循环

        model.train()  # 切换到训练模式
        watch_loss = []  # 用于记录每批次 loss

        for batch_index in range(train_sample // batch_size):  # 遍历所有 batch
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]  # 取 batch 输入
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]  # 取 batch 标签
            loss = model(x, y)  # 前向传播并计算损失
            loss.backward()  # 反向传播
            optim.step()  # 更新参数
            optim.zero_grad()  # 梯度清零
            watch_loss.append(loss.item())  # 记录 loss

        print("=======\n第%d轮平均loss：%f" % (epoch + 1, np.mean(watch_loss)))  # 输出平均损失
        acc = evaluate(model)  # 评估模型
        log.append([acc, float(np.mean(watch_loss))])  # 记录日志

    torch.save(model.state_dict(), 'model.bin')  # 保存模型参数
    print(log)  # 打印训练日志
    plt.plot(range(len(log)), [l[0] for l in log], label=['acc'])
    plt.plot(range(len(log)), [l[1] for l in log], label=['loss'])
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    input_size = 5  # 输入维度
    model = TorchModel(input_size)  # 创建模型
    model.load_state_dict(torch.load(model_path))  # 加载模型参数
    print(model.state_dict())  # 打印模型参数

    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 禁用梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 得到预测结果
    for vec, res in zip(input_vec, result):  # 遍历输入和预测
        print("输入：%s 预测类别：%d 概率值：%f" % (vec, round(float(res)), res))  # 打印预测信息


if __name__ == '__main__':  # 程序入口
    main()

    # 定义测试输入向量
    # test_vec = [
    #     [0.25886654, 0.35886654, 0.45886654, 0.75886654, 0.1588665],
    #     [0.25886654, 0.35886654, 0.45886654, 0.75886654, 0.7588665],
    #     [0.95886654, 0.35886654, 0.45886654, 0.75886654, 0.3588665],
    #     [0.00886654, 0.35886654, 0.45886654, 0.75886654, 0.0588665]
    # ]
    #
    # predict("model.bin", test_vec)  # 调用预测函数

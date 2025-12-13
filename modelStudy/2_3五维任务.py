# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ------------------------------------
# 1. 模型定义 (Model Definition)
# ------------------------------------

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()

        # 线性层：输入 5 维，输出 num_classes (即 5 维)
        self.linear = nn.Linear(input_size, num_classes)

        # 激活函数：使用 Softmax，PyTorch 的 CrossEntropyLoss 内部已包含 LogSoftmax
        # 所以在 forward 中可以省略 Softmax
        # 这里的 activation 和 loss 定义与原代码略有不同，以便使用 CrossEntropyLoss
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 定义前向传播过程
        y_pred = self.linear(x)  # 输出 logits，shape: (batch_size, 5)

        if y is not None:
            # CrossEntropyLoss 接受 logit 输出 (y_pred) 和类别索引 (y)
            return self.loss_func(y_pred, y)
        else:
            # 预测时返回 logit
            return y_pred

        # ------------------------------------


# 2. 数据集生成 (Dataset Generation)
# ------------------------------------

def build_sample():
    x = np.random.randn(5)  # 随机生成一个 5 维向量

    # 找到最大值的索引，即为正确类别 (0, 1, 2, 3, or 4)
    y_class = np.argmax(x)

    return x, y_class


def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y_class = build_sample()
        X.append(x)
        Y.append(y_class)

    # 优化后的转换：先转 NumPy 数组，再转 Tensor
    # 注意：CrossEntropyLoss 接受 LongTensor 格式的类别索引
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


# ------------------------------------
# 3. 评估函数 (Evaluation)
# ------------------------------------

def evaluate(model):
    model.eval()  # 切换到评估模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    # 统计每个类别的真实样本数
    true_counts = [int(sum(y == i)) for i in range(5)]
    print(f"本次预测集中各类别样本数 (0-4): {true_counts}")

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred_logits = model(x)

        # 预测结果：找到 5 个输出中最大的那个索引
        # shape: (100, 5) -> (100)
        y_pred_class = torch.argmax(y_pred_logits, dim=1)

        for y_p, y_t in zip(y_pred_class, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1

    acc = correct / (correct + wrong)
    print("正确预测个数：%d，正确率：%f" % (correct, acc))
    return acc


# ------------------------------------
# 4. 主训练函数 (Main Training)
# ------------------------------------

def main():
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每批次大小
    train_sample = 5000  # 训练样本数量
    input_size = 5  # 输入维度
    num_classes = 5  # 分类数
    learning_rate = 0.01  # 提高学习率以加快收敛

    model = TorchModel(input_size, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        # 遍历所有 batch (250次)
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]

            # 训练步骤
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        print(f"=======\n第{epoch + 1}轮平均loss：{np.mean(watch_loss):.4f}")
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), 'model_5cls.bin')
    print("\n训练日志:", log)

    # 绘图部分
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(log)), [l[0] for l in log], label='Accuracy')
    plt.plot(range(len(log)), [l[1] for l in log], label='Loss')
    plt.legend()
    plt.title('5-Class Classification Training Metrics')
    plt.xlabel('Epoch')  # X轴是训练轮次
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()

import numpy as np
import torch
import torch.nn as nn

# 手动实现交叉熵的计算
ce_loss = nn.CrossEntropyLoss()  # 内部运行了softmax
# 假设3个样本，每个都在做3分类，还没有归1化
pred = torch.FloatTensor([[0.3, 0.1, 0.3], [0.9, 0.2, 0.9], [0.5, 0.4, 0.2]])  # n*class_num
# 正确的类别分别1,2,0
target = torch.LongTensor([1, 2, 0])
# 等价于下面这个写法，归1化softmax
"""
[0,1,0]  softmax约等于 [0.354,0.290,0.354] 下面2个也要算出来
[0,0,1]
[1,0,0]

[0,1,0] -（0*log0.354+1*log0.290+0*log0.354）约等于1.238
[0,0,1]  -log0.401 约等于 0.915 
[1,0,0]  -log0.378  约等于 0.973
1.238 + 0.915 + 0.973 / 3 约等于1.042
"""
loss = ce_loss(pred, target)
print(loss, "torch交叉熵")


def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)


def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    for i, t in enumerate(target):
        one_hot_target[i][t] = 1
    return one_hot_target


def cross_entropy(pred, target):
    batch_size, class_num = pred.shape
    pred = softmax(pred)
    target = to_one_hot(target, pred.shape)
    entropy = -np.sum(target * np.log(pred), axis=1)
    return sum(entropy) / batch_size


print(cross_entropy(pred.numpy(), target.numpy()), "手动交叉熵")

print(np.log(2.7))  # 底数e

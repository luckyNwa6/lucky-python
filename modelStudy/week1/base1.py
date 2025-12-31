# 实际规律
import matplotlib.pyplot as pyplot

X = [0.01 * x for x in range(100)]
Y = [2 * x ** 2 + 3 * x + 4 for x in X]

# X = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
# Y = [4.0, 4.0302, 4.0608, 4.0918, 4.1232, 4.155, 4.1872, 4.2198, 4.2528, 4.2862]

print(len(X))
print(len(Y))


# 定义的模型
def func(x):
    y = w1 * x ** 2 + w2 * x + w3
    return y


# 损失函数
def loss(y_pred, y_true):
    return (y_pred - y_true) ** 2


# 推导 w1的梯度 带入本质是这个 (w1 * x ** 2 + w2 * x + w3 - y_true) ** 2
# 链式法则  (y_pred - y_true) ** 2 求导
# 看成2个内外层求导 外层 u **2  内层 u=y_pred - y_true 对 f`(w1 * x ** 2 + w2 * x + w3 - y_true)
# 求完导数剩 2u * x ** 2       因为w1是变量  比如 3w1求导 结果是3
# 再把u替换2(y_pred - y_true) * x ** 2

# w2 进行求导 那么其他w1、w3都是常量忽略不参与求导 w2*x进行求导
# 2u * x ==》  2 * (y_pred - y_true) * x

# w3是常量什么都没有了 只剩 2u

# 权重随机初始化
w1, w2, w3 = 1, 0, -1
# 学习率设置
lr = 0.1
# bash size
batch_size = 10

# 训练过程
for epoch in range(1000):
    epoch_loss = 0
    grad_w1 = 0
    grad_w2 = 0
    grad_w3 = 0
    counter = 0
    for x, y_true in zip(X, Y):
        y_pred = func(x)
        epoch_loss += loss(y_pred, y_true)
        counter += 1
        # 梯度计算
        grad_w1 += 2 * (y_pred - y_true) * x ** 2
        grad_w2 += 2 * (y_pred - y_true) * x
        grad_w3 += 2 * (y_pred - y_true)
        if counter == batch_size:
            # 权重更新
            w1 = w1 - lr * grad_w1 / batch_size  # sgd
            w2 = w2 - lr * grad_w2 / batch_size
            w3 = w3 - lr * grad_w3 / batch_size
            counter = 0
            grad_w1 = 0
            grad_w2 = 0
            grad_w3 = 0
    epoch_loss = epoch_loss / len(X)
    print("第%d轮，loss %f" % (epoch, epoch_loss))
    if epoch_loss < 0.00001:
        break
print(f"训练后权重：w1：{w1} w2：{w2} w3：{w3}")

Yp = [func(i) for i in X]

pyplot.scatter(X, Y, color='red')
pyplot.scatter(X, Yp)
pyplot.show()

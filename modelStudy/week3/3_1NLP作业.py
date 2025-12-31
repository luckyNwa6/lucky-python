# coding:utf8

import json
import random

import numpy as np
import torch
import torch.nn as nn

"""
语序相关任务，
输入6个字符串，做多分类，
生成的字符串必须包含a，a在哪个位置就是第几分类

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        # self.pool = nn.AvgPool1d(sentence_length)  # 池化层会乱序
        self.rnn = nn.RNN(input_size=vector_dim, hidden_size=vector_dim, batch_first=True)
        # self.classify = nn.Linear(vector_dim, 1)  # 线性层
        self.classify = nn.Linear(vector_dim, sentence_length)
        self.loss = nn.CrossEntropyLoss()  # 交叉熵
        # self.activation = torch.sigmoid  # sigmoid归一化函数
        # self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # x = x.transpose(1, 2)  # (batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        # x = self.pool(x)  # (batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        # x = x.squeeze()  # (batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        # x = self.classify(x)  # (batch_size, vector_dim) -> (batch_size, 1) 3*20 20*1 -> 3*1
        # y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        # (batch, 6, vector_dim)
        output, hidden = self.rnn(x)  # output每个位置隐藏状态 hidden最后一个时间步的隐藏状态
        h_last = hidden.squeeze(0)  # 模型最终只能靠“走到第 6 个字符时的记忆”来猜
        y_pred = self.classify(h_last)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 为每个字生成一个标号，只识别a，其他都是噪声

def build_vocab():
    vocab = {"pad": 0, "a": 1, "other": 2}
    return vocab


# 随机生成一个样本 ，除了a其他样本都没有意义，如果是原来那种噪音太大

def build_sample(vocab, sentence_length):
    # 1️⃣ 先构造一个全是“other”的序列  x = ["other", "other", "other", "a", "other", "other"]
    x = ["other"] * sentence_length

    # 2️⃣ 随机选一个位置（0 ~ sentence_length-1）
    pos = random.randint(0, sentence_length - 1)

    # 3️⃣ 在该位置放入 'a'
    x[pos] = "a"

    # 4️⃣ 标签就是 a 出现的位置
    y = pos

    # 5️⃣ 把字符序列映射成整数 ID

    x = [vocab[c] for c in x]  # x = ["other", "other", "other", "a", "other", "other"]-->[2,2,1,2]

    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    # print("创建的x", dataset_x)
    # print("创建的Y", dataset_y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(5, vocab, sentence_length)

    correct = 0
    with torch.no_grad():
        logits = model(x)  # (batch, 6)
        pred = torch.argmax(logits, dim=1)

        correct = (pred == y).sum().item()

    acc = correct / len(y)
    print(f"准确率: {acc:.4f}")
    return acc


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_sequences):
    char_dim = 20
    sentence_length = 6

    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x = []
    for seq in input_sequences:
        x.append([vocab[c] for c in seq])

    x = torch.LongTensor(x)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)

    for seq, p in zip(input_sequences, pred):
        print(f"输入: {seq} → 预测 a 在位置: {p.item()}")


if __name__ == "__main__":
    main()
    # test_strings = [["other", "a", "other", "other", "other", "other"]]
    # predict("model.pth", "vocab.json", test_strings)

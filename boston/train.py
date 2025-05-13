import paddle
import numpy as np
import matplotlib.pyplot as plt
import paddle.nn.functional as F



""" 数据录入
"""

paddle.set_default_dtype("float64")

data_file = "data.txt"
data = np.fromfile(data_file, sep=" ")
print(f"Origin shape: {data.shape}")

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTATA','MEDV']
feature_num = len(feature_names)

data = data.reshape((len(data) // feature_num, feature_num))
print(f"New shape: {data.shape}")



""" 划分集
"""

train_size = int(len(data) * 0.8)

train = data[:train_size]
test = data[train_size:]

print(f"The shape of TRAIN: {train.shape}")
print(f"The shape of TEST: {test.shape}")



""" 归一化
"""

for _ in range(feature_num):
    train_max = max(train[:, _])
    train_min = min(train[:, _])

    train[:, _] = [(data - train_min) / (train_max - train_min) for data in train[:, _]]
    test[:, _] = [(data - train_min) / (train_max - train_min) for data in test[:, _]]

print("Data normalizated.")



""" 模型定义
"""

class Regressor(paddle.nn.Layer):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc = paddle.nn.Linear(in_features=13, out_features=1)

    def forward(self, inputs):
        x = self.fc(inputs)
        return x

model = Regressor()



""" 模型训练
"""

model.train()
optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

# 设置超参数
EPOCH_NUM = 10
BATCH_SIZE = 20

# 存储训练次数和每次的损失值
Iter_train = 0
Iter_list = []
Trainloss_list = []

# 开始训练
for epoch_id in range(EPOCH_NUM):
    np.random.shuffle(train)

    # 拆分batch
    mini_batches = [train[k:k + BATCH_SIZE] for k in range(0, len(train), BATCH_SIZE)]
    if epoch_id == 0:
        print(f"The num of mini-batch: {len(mini_batches)}")
        print(f"The shape of the first mini-batch: {mini_batches[0].shape}")

    # 训练batch
    for iter_id, mini_batch in enumerate(mini_batches):
        Iter_train += 1

        # 划分变量
        x = np.array(mini_batch[:, :-1])
        y = np.array(mini_batch[:, -1:])

        # 计算张量
        house_features = paddle.to_tensor(x)
        prices = paddle.to_tensor(y)

        # 前向计算, 计算损失
        preds = model(house_features)
        loss = F.square_error_cost(preds, prices)
        mean_loss = paddle.mean(loss)

        # 打印损失值
        if iter_id % 10 == 0:
            print(f"eoch_id: {epoch_id}, Iter_train: {Iter_train}, avg_loss: {float(mean_loss)}")

        # 存储训练次数和损失值
        Iter_list.append(Iter_train)
        Trainloss_list.append(float(mean_loss))

        # 反向传播, 更新梯度
        mean_loss.backward()
        optimizer.step()
        optimizer.clear_grad()



""" 损失可视化
"""

plt.plot(Iter_list, Trainloss_list, color="#aa00ff", alpha=0.5)
plt.title("Loss vs Iter times")
plt.xlabel("Iter times")
plt.ylabel("Loss")
plt.show()



""" 模型保存
"""

paddle.save(model.state_dict(), "LR_model.pdparams")
print("Model saved.")




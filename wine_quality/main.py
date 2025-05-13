import paddle
import numpy as np
import matplotlib.pyplot as plt
import paddle.nn.functional as F



""" 数据录入
"""

paddle.set_default_dtype("float64")

data_file = "winequality-red.csv"
data = np.loadtxt(data_file, delimiter=";", skiprows=1) 
print(f"Origin shape: {data.shape}")

feature_names = feature_names = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'quality'
]
feature_num = len(feature_names)



""" 划分集
"""

train_size = int(len(data) * 0.8)

train = data[:train_size]
test = data[train_size:]

print(f"The shape of TRAIN: {train.shape}")
print(f"The shape of TEST: {test.shape}")



""" 归一化
"""

train_min, train_max = np.zeros(feature_num), np.zeros(feature_num)
for _ in range(feature_num):
    train_min[_] = np.min(train[:, _])
    train_max[_] = np.max(train[:, _])
    train[:, _] = (train[:, _] - train_min[_]) / (train_max[_] - train_min[_])
    test[:, _] = (test[:, _] - train_min[_]) / (train_max[_] - train_min[_])

print("Data normalizated.")



""" 模型定义
"""

class WineQualityModel(paddle.nn.Layer):
    def __init__(self):
        super(WineQualityModel, self).__init__()
        self.fc1 = paddle.nn.Linear(11, 64)
        self.fc2 = paddle.nn.Linear(64, 32)
        self.fc3 = paddle.nn.Linear(32, 1)
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = WineQualityModel()



""" 模型训练
"""

model.train()
optimizer = paddle.optimizer.SGD(learning_rate=0.005, parameters=model.parameters())

# 设置超参数
EPOCH_NUM = 100
BATCH_SIZE = 32

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
        x_tensor = paddle.to_tensor(x)
        y_tensor = paddle.to_tensor(y)

        # 前向计算, 计算损失
        preds = model(x_tensor)
        loss = F.square_error_cost(preds, y_tensor)
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



""" 模型加载
"""

model_dict = paddle.load("LR_model.pdparams")
model.load_dict(model_dict)
model.eval()

print("Model loaded.")


""" 模型测试
"""

True_Price = []
Pred_Price = []

for test_sample in test:
    # 测试及前向传播
    x_test = test_sample[:-1].reshape(1, -1)
    label_test = test_sample[-1]
    x_test = paddle.to_tensor(x_test)

    # 前向传播的结果反归一化处理
    pred = model(x_test).numpy().item()
    pred = pred * (train_max[-1] - train_min[-1]) + train_min[-1]
    label_test = label_test * (train_max[-1] - train_min[-1]) + train_min[-1]
    Pred_Price.append(pred)
    True_Price.append(label_test)


# 打印结果
# for _ in range(len(test)):
#     print(f"Pred: {Pred_Price[_]} \t True: {True_Price[_]} \t R: {Pred_Price[_] - True_Price[_]}")


# 绘制散点图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 预测值, 真实值与组数
x_datas = np.arange(test.shape[0])
ax1.scatter(x_datas, Pred_Price, label="Pred_Price", color="#00aaff")
ax1.scatter(x_datas, True_Price, label="True_Price", color="#88ff00")

ax1.set_xlabel("k")
ax1.set_ylabel("Quality")
ax1.set_title("True vs Pred Quality")
ax1.legend()


# 预测值与真实值
ax2.scatter(True_Price, Pred_Price, color="#aa00ff", alpha=0.5)

min_val = min(min(True_Price), min(Pred_Price))
max_val = max(max(True_Price), max(Pred_Price))
ax2.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label="R=0")

ax2.set_xlabel("True Quality")
ax2.set_ylabel("Pred Quality")
ax2.set_title("True vs Pred Quality")
ax2.legend()


plt.tight_layout()
plt.show()




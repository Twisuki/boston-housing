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

for itr, test_sample in enumerate(test):
    # 测试机前向传播
    x_test = test_sample[:-1]
    label_test = test_sample[-1]
    x_test = paddle.to_tensor(x_test)

    # 前向传播的结果反归一化处理
    pred_test = model(x_test)
    Pred_Price.append(pred_test.numpy().item())
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
ax1.set_ylabel("House Price (MEDV)")
ax1.set_title("True vs Pred Prices")
ax1.legend()


# 预测值与真实值
ax2.scatter(True_Price, Pred_Price, color="#aa00ff", alpha=0.5)

min_val = min(min(True_Price), min(Pred_Price))
max_val = max(max(True_Price), max(Pred_Price))
ax2.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label="R=0")

ax2.set_xlabel("True Price")
ax2.set_ylabel("Pred Price")
ax2.set_title("True vs Pred Prices")
ax2.legend()


plt.tight_layout()
plt.show()




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



""" 模型加载
"""

model_dict = paddle.load("LR_model.pdparams")
model.load_dict(model_dict)
model.eval()

print("Model loaded.")


""" 模型测试
"""

True_Quality = []
Pred_Quality = []

for test_sample in test:
    # 测试及前向传播
    x_test = test_sample[:-1].reshape(1, -1)
    label_test = test_sample[-1]
    x_test = paddle.to_tensor(x_test)

    # 前向传播的结果反归一化处理
    pred = model(x_test).numpy().item()
    pred = pred * (train_max[-1] - train_min[-1]) + train_min[-1]
    label_test = label_test * (train_max[-1] - train_min[-1]) + train_min[-1]
    Pred_Quality.append(pred)
    True_Quality.append(label_test)


# 打印结果
# for _ in range(len(test)):
#     print(f"Pred: {Pred_Quality[_]} \t True: {True_Quality[_]} \t R: {Pred_Quality[_] - True_Quality[_]}")


# 绘制散点图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 预测值, 真实值与组数
x_datas = np.arange(test.shape[0])
ax1.scatter(x_datas, Pred_Quality, label="Pred_Quality", color="#00aaff")
ax1.scatter(x_datas, True_Quality, label="True_Quality", color="#88ff00")

ax1.set_xlabel("k")
ax1.set_ylabel("Quality")
ax1.set_title("True vs Pred Quality")
ax1.legend()


# 预测值与真实值
ax2.scatter(True_Quality, Pred_Quality, color="#aa00ff", alpha=0.5)

min_val = min(min(True_Quality), min(Pred_Quality))
max_val = max(max(True_Quality), max(Pred_Quality))
ax2.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label="R=0")

ax2.set_xlabel("True Quality")
ax2.set_ylabel("Pred Quality")
ax2.set_title("True vs Pred Quality")
ax2.legend()


plt.tight_layout()
plt.show()




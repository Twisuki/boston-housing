import paddle
import numpy as np
import matplotlib.pyplot as plt
import paddle.nn.functional as F



""" 数据录入
"""

paddle.set_default_dtype("float64")

# 加载MNIST数据集
transform = paddle.vision.transforms.Compose([
    paddle.vision.transforms.ToTensor(),
    paddle.vision.transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

# 转换为DataLoader
BATCH_SIZE = 64
train_loader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = paddle.io.DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")




""" 模型定义
"""

class DigitRecognizer(paddle.nn.Layer):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        # CNN网络结构
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = paddle.nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.max_pool = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.fc1 = paddle.nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.fc2 = paddle.nn.Linear(in_features=128, out_features=10)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = paddle.flatten(x, start_axis=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = DigitRecognizer()



""" 模型训练
"""

model.train()
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

# 设置超参数
EPOCH_NUM = 5
BATCH_SIZE = 64

# 存储训练次数和每次的损失值
Iter_train = 0
Iter_list = []
Trainloss_list = []

# 创建数据加载器
train_loader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 开始训练
for epoch_id in range(EPOCH_NUM):
    # 训练batch
    for batch_id, (x, y) in enumerate(train_loader()):
        Iter_train += 1

        # 计算张量
        x_tensor = x.astype('float64')
        y_tensor = y.astype('int64')

        # 前向计算, 计算损失
        preds = model(x_tensor)
        loss = F.cross_entropy(preds, y_tensor)

        # 打印损失值
        if batch_id % 100 == 0:
            print(f"epoch_id: {epoch_id}, batch_id: {batch_id}, loss: {float(loss)}")

        # 存储训练次数和损失值
        Iter_list.append(Iter_train)
        Trainloss_list.append(float(loss))

        # 反向传播, 更新梯度
        loss.backward()
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

paddle.save(model.state_dict(), "DigitRecognizer.pdparams")
print("Model saved.")



""" 模型加载
"""

model_dict = paddle.load("DigitRecognizer.pdparams")
model.load_dict(model_dict)
model.eval()

print("Model loaded.")


""" 模型测试
"""

# 创建测试数据加载器
test_loader = paddle.io.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 存储预测结果
Pred_Labels = []
True_Labels = []

# 测试集评估
for batch_id, (x, y) in enumerate(test_loader()):
    x_tensor = x.astype('float64')
    y_tensor = y.astype('int64')
    
    # 前向传播
    preds = model(x_tensor)
    
    # 获取预测结果
    Pred_Labels.append(paddle.argmax(preds, axis=1).numpy())
    True_Labels.append(y_tensor.numpy())



# 绘制散点图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 预测值, 真实值与组数
x_datas = np.arange(test_dataset.shape[0])
ax1.scatter(x_datas, Pred_Labels, label="Pred_Labels", color="#00aaff")
ax1.scatter(x_datas, True_Labels, label="True_Labels", color="#88ff00")

ax1.set_xlabel("k")
ax1.set_ylabel("Labels")
ax1.set_title("True vs Pred Labels")
ax1.legend()


# 预测值与真实值
ax2.scatter(True_Labels, Pred_Labels, color="#aa00ff", alpha=0.5)

min_val = min(min(True_Labels), min(Pred_Labels))
max_val = max(max(True_Labels), max(Pred_Labels))
ax2.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label="R=0")

ax2.set_xlabel("True Labels")
ax2.set_ylabel("Pred Labels")
ax2.set_title("True vs Pred Labels")
ax2.legend()


plt.tight_layout()
plt.show()








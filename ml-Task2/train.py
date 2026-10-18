
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_preprocess import CatDogDataset
from cnn_model import CatDogCNN


# 1. 设置数据集路径
train_dir = "dataset/training_set"
test_dir = "dataset/test_set"

# 2. 加载数据
train_data = CatDogDataset(__import__("pathlib").Path(train_dir))
test_data = CatDogDataset(__import__("pathlib").Path(test_dir))

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

print("训练集图片数量：", len(train_data))
print("测试集图片数量：", len(test_data))


# 3. 创建模型
model = CatDogCNN()

# 4. 定义损失函数和优化器
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 5. 训练模型
for epoch in range(5):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        labels = labels.float().view(-1, 1)

        # 预测
        outputs = model(images)

        # 计算误差
        loss = loss_function(outputs, labels)

        # 计算梯度
        optimizer.zero_grad()
        loss.backward()

        # 更新模型参数
        optimizer.step()

        total_loss += loss.item()

    print("第", epoch + 1, "轮训练，平均批次损失：",
          total_loss / len(train_loader))


# 6. 保存模型
torch.save(model.state_dict(), "catdog_model.pth")
print("模型保存成功！")


# 7. 测试模型
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        # 分数大于等于 0，预测为狗；否则预测为猫
        predictions = (outputs >= 0).long()
        labels = labels.view(-1, 1)

        correct += (predictions == labels).sum().item()
        total += labels.numel()

print("测试集准确率：", correct / total)

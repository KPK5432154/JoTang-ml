
from pathlib import Path
import random

import torch
import matplotlib.pyplot as plt

from data_preprocess import CatDogDataset
from cnn_model import CatDogCNN


# 1. 设置路径
BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "dataset" / "test_set"
MODEL_PATH = BASE_DIR / "catdog_model.pth"

# 2. 加载测试集
test_data = CatDogDataset(TEST_DIR)

# 3. 加载训练好的模型
model = CatDogCNN()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)
model.eval()

# 4. 随机选择两张测试图片
random.seed(42)
indices = random.sample(range(len(test_data)), 2)

# 5. 显示原图和特征图
fig, axes = plt.subplots(2, 9, figsize=(18, 7))

for row in range(2):
    # 取出一张图片和标签
    image, label = test_data[indices[row]]

    # 增加一个批次维度
    image = image.unsqueeze(0)

    # 只计算，不更新模型参数
    with torch.no_grad():
        # 第一层卷积
        x = model.features[0](image)

        # 经过 ReLU 激活
        x = model.features[1](x)

    # 取出这张图片的 16 个特征图
    feature_maps = x[0]

    # 显示原图
    original = image[0].permute(1, 2, 0).numpy()

    # 归一化
    if original.max() > 1:
        original = original / 255.0

    name = "猫" if label == 0 else "狗"

    axes[row, 0].imshow(original)
    axes[row, 0].set_title("original:" + name)
    axes[row, 0].axis("off")

    # 显示前 8 个通道
    for channel in range(8):
        feature = feature_maps[channel].numpy()

        axes[row, channel + 1].imshow(feature, cmap="gray")
        axes[row, channel + 1].set_title("channel " + str(channel))
        axes[row, channel + 1].axis("off")

plt.tight_layout()
plt.show()

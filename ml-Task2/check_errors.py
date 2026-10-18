
from pathlib import Path

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

wrong_images = []
correct = 0

# 4. 逐张检查图片
with torch.no_grad():
    for i in range(len(test_data)):
        image, label = test_data[i]

        # 增加批次维度
        image = image.unsqueeze(0)

        # 模型预测
        output = model(image)
        probability = torch.sigmoid(output).item()

        if probability >= 0.5:
            prediction = 1
        else:
            prediction = 0

        # 统计正确数量
        if prediction == label:
            correct += 1
        else:
            wrong_images.append(
                (i, label, prediction, probability)
            )

# 5. 输出统计结果
accuracy = correct / len(test_data)

print("测试集总数：", len(test_data))
print("预测正确：", correct)
print("预测错误：", len(wrong_images))
print("测试集准确率：", f"{accuracy:.2%}")

# 6. 显示最多 12 张预测错误的图片
if len(wrong_images) == 0:
    print("没有发现预测错误的图片。")
else:
    count = min(12, len(wrong_images))

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    for j in range(count):
        index, label, prediction, probability = wrong_images[j]

        image, _ = test_data[index]

        # 转换成 matplotlib 能显示的格式
        image = image.permute(1, 2, 0).numpy()

        # 如果像素范围是 0~255，转换到 0~1
        if image.max() > 1:
            image = image / 255.0

        true_name = "Cat" if label == 0 else "Dog"
        pred_name = "Cat" if prediction == 0 else "Dog"

        axes[j].imshow(image)
        axes[j].set_title(
            f"True: {true_name}\n"
            f"Pred: {pred_name}\n"
            f"Dog score: {probability:.2f}"
        )
        axes[j].axis("off")

        print(
            f"样本索引：{index} | "
            f"真实类别：{true_name} | "
            f"预测类别：{pred_name} | "
            f"狗的预测分数：{probability:.3f}"
        )

    # 隐藏没有使用的图表位置
    for j in range(count, 12):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

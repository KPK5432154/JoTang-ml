
from pathlib import Path
import argparse

import torch
import matplotlib.pyplot as plt
from PIL import Image

from cnn_model import CatDogCNN
from data_preprocess import preprocess_image


# 当前文件所在目录
BASE_DIR = Path(__file__).resolve().parent

# 已训练好的模型文件
MODEL_PATH = BASE_DIR / "catdog_model.pth"

# 处理后图片的保存位置
OUTPUT_PATH = BASE_DIR / "processed_image.png"


def predict_image(image_path):
    image_path = Path(image_path)

    # 1. 检查文件是否存在
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到输入图片：{image_path}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"找不到训练好的模型：{MODEL_PATH}\n"
            "请确认 catdog_model.pth 存在。"
        )

    # 2. 加载训练好的模型参数
    print(f"正在加载模型：{MODEL_PATH}")

    model = CatDogCNN()

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("训练好的模型加载成功！")

    # 3. 读取图片，并使用训练时相同的预处理
    with Image.open(image_path) as img:
        original_image = img.convert("RGB")

    processed_tensor = preprocess_image(original_image)

    # 原来是 [3, 128, 128]
    # 增加批次维度后变为 [1, 3, 128, 128]
    input_tensor = processed_tensor.unsqueeze(0)

    print("模型输入形状：", tuple(input_tensor.shape))

    # 4. 使用训练好的模型进行推理
    with torch.no_grad():
        logits = model(input_tensor)
        dog_probability = torch.sigmoid(logits).item()

    cat_probability = 1.0 - dog_probability

    if dog_probability >= 0.5:
        predicted_class = "dog"
        predicted_probability = dog_probability
    else:
        predicted_class = "cat"
        predicted_probability = cat_probability

    # 5. 保存预处理后的图片
    processed_image = processed_tensor.permute(1, 2, 0).numpy()
    processed_image = (processed_image * 255).round().clip(0, 255).astype("uint8")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(processed_image).save(OUTPUT_PATH)

    # 6. 输出预测结果
    print("\n========== 预测结果 ==========")
    print(f"输入图片：{image_path}")
    print(f"预测类别：{predicted_class}")
    print(f"预测类别概率：{predicted_probability:.2%}")
    print(f"猫的预测概率：{cat_probability:.2%}")
    print(f"狗的预测概率：{dog_probability:.2%}")
    print(f"处理后的图片：{OUTPUT_PATH}")

    # 7. 展示原图和处理后的图片
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(processed_image)
    axes[1].set_title(
        f"Prediction: {predicted_class}\n"
        f"Probability: {predicted_probability:.2%}"
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用训练好的 CNN 模型识别猫和狗"
    )

    parser.add_argument(
        "image",
        help="需要预测的本地图片路径",
    )

    args = parser.parse_args()
    predict_image(args.image)


from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps


# 1. 基本配置
DATA_DIR = Path(
    r"E:\C-learning\JoTang-ml\ml-Task2\dataset\training_set"
)

IMAGE_SIZE = 128
BATCH_SIZE = 8
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# 2. 图片预处理：训练和预测共用
def preprocess_image(img):
    # 统一为 RGB 三通道
    img = img.convert("RGB")

    # 保持宽高比，缩放后补边
    img = ImageOps.pad(
        img,
        (IMAGE_SIZE, IMAGE_SIZE),
        method=Image.Resampling.LANCZOS,
        color=(128, 128, 128),
    )

    # 将像素从 0~255 归一化到 0~1
    array = np.array(img, dtype=np.float32) / 255.0

    # [H, W, C] 转换为 [C, H, W]
    image = torch.from_numpy(array).permute(2, 0, 1)

    return image


# 3. 定义数据集
class CatDogDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []

        # 猫的标签为 0，狗的标签为 1
        for name, label in [("cats", 0), ("dogs", 1)]:
            folder = self.root_dir / name

            if not folder.exists():
                raise FileNotFoundError(
                    f"找不到文件夹：{folder}"
                )

            for path in sorted(folder.iterdir()):
                if (
                    path.is_file()
                    and path.suffix.lower() in IMAGE_EXTS
                ):
                    self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError(
                f"没有在 {self.root_dir} 中找到图片，请检查数据路径和格式。"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as img:
            image = preprocess_image(img)

        return image, label


# 4. 随机展示预处理前后的图片
def show_images(dataset, count=4):
    indices = random.sample(
        range(len(dataset)),
        min(count, len(dataset)),
    )

    fig, axes = plt.subplots(
        len(indices), 2,
        figsize=(8, 4 * len(indices)),
    )

    axes = np.atleast_2d(axes)

    for row, index in enumerate(indices):
        path, label = dataset.samples[index]

        # 原始图片
        with Image.open(path) as img:
            original = img.convert("RGB").copy()

        # 预处理后的图片
        processed, _ = dataset[index]
        processed = processed.permute(1, 2, 0).numpy()

        name = "Cat" if label == 0 else "Dog"

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(
            f"Before: {name} ({original.width}x{original.height})"
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(processed)
        axes[row, 1].set_title(
            f"After: {name} ({IMAGE_SIZE}x{IMAGE_SIZE})"
        )
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.show()


# 5. 主程序
def main():
    dataset = CatDogDataset(DATA_DIR)

    print("图片总数：", len(dataset))
    print(
        "猫的数量：",
        sum(label == 0 for _, label in dataset.samples),
    )
    print(
        "狗的数量：",
        sum(label == 1 for _, label in dataset.samples),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    images, labels = next(iter(loader))

    print("\n===== Batch 信息 =====")
    print("图片形状：", images.shape)
    print("标签形状：", labels.shape)
    print("像素最小值：", images.min().item())
    print("像素最大值：", images.max().item())
    print("标签示例：", labels.tolist())

    show_images(dataset)


if __name__ == "__main__":
    main()

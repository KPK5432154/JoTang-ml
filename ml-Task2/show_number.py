
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image


# 1. 设置数据集路径
DATA_DIR = Path(r"E:\C-learning\JoTang-ml\ml-Task2\dataset\training_set")

# 2. 设置随机数种子，让每次抽样结果尽量一致
random.seed(42)


def main():
    # 3. 获取猫和狗的图片
    cat_folder = DATA_DIR / "cats"
    dog_folder = DATA_DIR / "dogs"

    cat_images = []
    dog_images = []

    # 获取猫的图片
    for path in cat_folder.iterdir():
        if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            cat_images.append(path)

    # 获取狗的图片
    for path in dog_folder.iterdir():
        if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            dog_images.append(path)

    # 4. 统计图片数量
    print("===== 数据集统计 =====")
    print("猫的图片数量：", len(cat_images))
    print("狗的图片数量：", len(dog_images))
    print("图片总数：", len(cat_images) + len(dog_images))

    # 5. 每类随机选择最多 4 张图片
    cat_samples = random.sample(cat_images, min(4, len(cat_images)))
    dog_samples = random.sample(dog_images, min(4, len(dog_images)))

    # 把图片路径和类别放在一起
    samples = []

    for path in cat_samples:
        samples.append((path, "猫"))

    for path in dog_samples:
        samples.append((path, "狗"))

    # 打乱图片顺序
    random.shuffle(samples)

    # 6. 查看图片信息
    print("\n===== 图片信息 =====")

    for path, label in samples:
        img = Image.open(path)

        print("类别：", label)
        print("文件名：", path.name)
        print("图片尺寸：", img.width, "x", img.height)
        print("颜色模式：", img.mode)
        print()

        img.close()

    # 7. 显示图片
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for i in range(len(samples)):
        path, label = samples[i]

        img = Image.open(path)
        img = img.convert("RGB")

        # 找到第 i 个显示位置
        row = i // 4
        col = i % 4

        axes[row, col].imshow(img)
        axes[row, col].set_title(label)
        axes[row, col].axis("off")

        img.close()

    # 如果图片不足 8 张，隐藏多余的位置
    for i in range(len(samples), 8):
        row = i // 4
        col = i % 4
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

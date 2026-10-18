
from pathlib import Path
from collections import Counter
import random

import matplotlib.pyplot as plt
from PIL import Image

# 数据集路径
DATA_DIR = Path(
    r"E:\C-learning\JoTang-ml\ml-Task2\dataset\training_set"
)

# 支持的图片格式
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(42)


def get_images(folder):
    """获取指定类别文件夹中的图片路径。"""
    if not folder.exists():
        print(f"目录不存在：{folder}")
        return []

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main():
    cats = get_images(DATA_DIR / "cats")
    dogs = get_images(DATA_DIR / "dogs")

    # 1. 统计数量
    print("===== 数据集统计 =====")
    print(f"猫的图片数量：{len(cats)}")
    print(f"狗的图片数量：{len(dogs)}")
    print(f"图片总数：{len(cats) + len(dogs)}")

    if not cats or not dogs:
        print("\n请检查目录名称是否为 cats 和 dogs。")
        return

    # 2. 随机抽样
    samples = [
        (p, "猫") for p in random.sample(cats, min(4, len(cats)))
    ] + [
        (p, "狗") for p in random.sample(dogs, min(4, len(dogs)))
    ]

    random.shuffle(samples)

    # 3. 检查图片尺寸与模式
    dimensions = Counter()
    modes = Counter()

    print("\n===== 随机样本信息 =====")

    for path, label in samples:
        try:
            with Image.open(path) as img:
                dimensions[img.size] += 1
                modes[img.mode] += 1
                print(
                    f"{label} | {path.name} | "
                    f"尺寸={img.width}x{img.height} | "
                    f"模式={img.mode}"
                )
        except Exception as e:
            print(f"无法读取：{path.name}，原因：{e}")

    # 4. 随机展示图片
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for ax, (path, label) in zip(axes.flat, samples):
        try:
            with Image.open(path) as img:
                ax.imshow(img.convert("RGB"))
                ax.set_title(
                    f"{label}: {path.name}\n{img.width}x{img.height}"
                )
                ax.axis("off")
        except Exception as e:
            ax.set_title(f"读取失败\n{path.name}")
            ax.axis("off")
            print(f"显示失败：{path.name}，原因：{e}")

    plt.tight_layout()
    plt.show()

    print("\n===== 样本尺寸分布 =====")
    for size, count in dimensions.most_common():
        print(f"{size[0]}x{size[1]}：{count} 张")

    print("\n===== 样本颜色模式 =====")
    for mode, count in modes.items():
        print(f"{mode}：{count} 张")


if __name__ == "__main__":
    main()

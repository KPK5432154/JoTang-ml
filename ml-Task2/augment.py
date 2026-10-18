
from pathlib import Path
import random
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageEnhance


#路径
DATA_DIR = Path(r"E:\C-learning\JoTang-ml\ml-Task2\dataset\test_set")

IMAGE_SIZE = 128
random.seed(42)


#读取一张图片并调整尺寸
def load_image(path):
    img = Image.open(path).convert("RGB")

    # 保留原始宽高比例，补齐到正方形
    img = ImageOps.pad(
        img,
        (IMAGE_SIZE, IMAGE_SIZE),
        color=(128, 128, 128)
    )

    return img


# 水平翻转
def flip_image(img):
    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    return img


# 4. 随机裁剪
def crop_image(img):
    width, height = img.size

    # 随机选择原图较短边的 70% 到 100%
    crop_size = random.randint(
        int(min(width, height) * 0.7),
        min(width, height)
    )

    # 随机选择裁剪位置
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)

    img = img.crop((
        left,
        top,
        left + crop_size,
        top + crop_size
    ))

    # 调整回统一尺寸
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    return img


# 随机颜色扰动
def color_image(img):
    # 随机改变亮度
    brightness = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # 随机改变对比度
    contrast = random.uniform(0.7, 1.3)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # 随机改变颜色饱和度
    color = random.uniform(0.7, 1.3)
    img = ImageEnhance.Color(img).enhance(color)

    return img


def main():
    #从猫和狗的测试集中各选一张图片
    cat_folder = DATA_DIR / "cats"
    dog_folder = DATA_DIR / "dogs"

    cat_images = list(cat_folder.glob("*.jpg"))
    dog_images = list(dog_folder.glob("*.jpg"))

    paths = [
        (random.choice(cat_images), "Cat"),
        (random.choice(dog_images), "Dog")
    ]

    # 显示原图和三种增强结果
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))

    titles = [
        "Original",
        "Horizontal Flip",
        "Random Crop",
        "Color Jitter"
    ]

    for row in range(2):
        path, label = paths[row]
        original = load_image(path)

        # 每种方法都从同一张原图开始
        images = [
            original,
            flip_image(original.copy()),
            crop_image(original.copy()),
            color_image(original.copy())
        ]

        for col in range(4):
            axes[row, col].imshow(images[col])
            axes[row, col].set_title(
                label + " - " + titles[col]
            )
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
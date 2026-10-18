import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#读取图片
img = Image.open("./png/outer1.png")
image = np.asarray(img, dtype=np.float32)

def conv2d(image,kernel,padding="valid"):
    
    KH,KW=kernel.shape
    if padding =="same":   #
        pad_h=KH//2    #整除
        pad_w=KW//2
        if image.ndim == 2:
            image = np.pad(
                image,
                ((pad_h, pad_h), (pad_w, pad_w)), #上下左右各补多少行
            )
        else:
            image = np.pad(
                image,
                ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            )


    if image.ndim ==2:

        H,W=image.shape
        out_H=H-KH+1
        out_W=W-KW+1   #移动次数+1即卷积得到的矩阵长宽

        output=np.zeros((out_H,out_W))  # 初始设置为0

        for i in range(out_H):
            for j in range (out_W):
                region =image[i:i+KH,j:j+KW] #3x3
                output[i,j]=np.sum(region*kernel)

        return output
    elif image.ndim ==3:
        H,W,C=image.shape

        out_H = H - KH + 1
        out_W = W - KW + 1

        output = np.zeros((out_H, out_W, C), dtype=np.float32)
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    region = image[i:i+KH, j:j+KW, c]
                    output[i, j, c] = np.sum(region * kernel)
        return output
    else :
        print("输入错误！")



#以下使用ai辅助完成
#  均值模糊：3×3
mean_kernel = np.ones((3, 3), dtype=np.float32) / 9
mean_result = conv2d(image, mean_kernel)


# 高斯模糊：5×5
# 使用常见的二项式近似高斯核，所有元素之和为 256
gaussian_kernel = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
], dtype=np.float32) / 256

gaussian_result = conv2d(image, gaussian_kernel)


#  锐化：3×3
sharpen_kernel = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
], dtype=np.float32)

sharpen_result = conv2d(image, sharpen_kernel)


# Sobel 边缘检测
#  RGB 转成灰度图，再检测边缘
gray = (
    0.299 * image[:, :, 0]
    + 0.587 * image[:, :, 1]
    + 0.114 * image[:, :, 2]
)

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

gx = conv2d(gray, sobel_x)
gy = conv2d(gray, sobel_y)

# 合并水平、垂直方向的梯度
edge_result = np.sqrt(gx ** 2 + gy ** 2)


# 输出形状，方便记录实验结果
print("\n卷积结果形状：")
print("均值模糊:", mean_result.shape)
print("高斯模糊:", gaussian_result.shape)
print("锐化:", sharpen_result.shape)
print("Sobel-X:", gx.shape)
print("Sobel-Y:", gy.shape)
print("边缘图:", edge_result.shape)


#  显示原图和四种结果
# 处理后的 RGB 数值可能超出 [0, 255]，显示前进行裁剪
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].imshow(image.astype(np.uint8))
axes[0, 0].set_title("Original RGB")

axes[0, 1].imshow(np.clip(mean_result, 0, 255).astype(np.uint8))
axes[0, 1].set_title("Mean Blur (3x3)")

axes[0, 2].imshow(np.clip(gaussian_result, 0, 255).astype(np.uint8))
axes[0, 2].set_title("Gaussian Blur (5x5)")

axes[1, 0].imshow(np.clip(sharpen_result, 0, 255).astype(np.uint8))
axes[1, 0].set_title("Sharpen")

axes[1, 1].imshow(edge_result, cmap="gray")
axes[1, 1].set_title("Sobel Edge Magnitude")

# 展示 Sobel-X 和 Sobel-Y，便于观察方向差异
axes[1, 2].imshow(np.hypot(gx, gy), cmap="gray")
axes[1, 2].set_title("Sobel X + Y")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()


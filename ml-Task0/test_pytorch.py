import sys
import torch

# 输出 Python 版本
print("Python 版本：", sys.version.split()[0])

# 输出 PyTorch 版本
print("PyTorch 版本：", torch.__version__)

# 创建一个张量
x = torch.tensor([1, 2, 3])
print("原始张量：", x)

# 张量运算
y = x * 2
print("运算结果：", y)
from pathlib import Path
import torch

from cnn_model import CatDogCNN

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "catdog_model.pth"

model = CatDogCNN()

# 加载之前训练好的参数
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)

# 切换到预测模式
model.eval()

print("已加载训练好的模型")
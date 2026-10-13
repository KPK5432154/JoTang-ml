import torch
import numpy as np
import matplotlib.pyplot as plt

from model import MLP
from dataset import get_data 

#传入模型
model=MLP()

model.load_state_dict(torch.load("mlp.pth"))

#传入数据
X_train,X_val,X_test,y_train,y_val,y_test=get_data()

#画坐标轴与点
xx,yy=np.meshgrid(np.linspace(-2,3,300),np.linspace(-2,2,300))
grid=np.c_[xx.ravel(),yy.ravel()]
grid=torch.tensor(grid,dtype=torch.float32)
#预测

with torch.no_grad():
    output=model(grid)
    pred=(output>0).float()

pred=pred.reshape(xx.shape)
#画预测区域
plt.contourf(xx,yy,pred,alpha=0.3)

#画数据点  

plt.scatter(X_test[:,0],X_test[:,1],c=y_test)
plt.xlabel("x1")
plt.ylabel("x2")

plt.title("decision boundry")
plt.show()
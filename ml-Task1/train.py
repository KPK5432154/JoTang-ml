import torch
import torch.nn as nn
import numpy as np

import time

start=time.time()

from dataset import get_data
from model import MLP  
#固定神经网络参数初始化
torch.manual_seed(42)

model=MLP()  
X_train,X_val,X_test,y_train,y_val,y_test=get_data()  

# 转换为张量
X_train=torch.tensor(X_train,dtype=torch.float32)
X_val=torch.tensor(X_val,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.float32)

# 训练  

criterion=nn.BCEWithLogitsLoss() #bce损失函数
optimizer=torch.optim.Adam(
    model.parameters(), #参数
    lr=0.001 #学习率 
    #lr=0.01
) #      优化器

times=100
loss_history=[]
acc_history=[]

for i in range (times):
    output=model(X_train) #传入数据
    pred=(output>0).float()

    loss=criterion(output,y_train.reshape(-1,1)) #计算损失
    acc=(pred==y_train.reshape(-1,1)).float().mean()

    optimizer.zero_grad() #清空梯度

    loss.backward()        #计算梯度

    optimizer.step()        #更新梯度

    loss_history.append(loss.item()) # 保存loss
    acc_history.append(acc.item())

    print(i,loss.item())        #打印次数和loss

torch.save(
    model.state_dict(),
    "mlp.pth"
)

end=time.time()
print("",end-start)

np.save("loss.npy",loss_history)
np.save("acc.npy",acc_history)
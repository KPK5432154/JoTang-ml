import torch

from model import MLP
from dataset import get_data 

from sklearn.metrics import confusion_matrix  #confusion

import numpy as np

model=MLP() 

model.load_state_dict(torch.load("mlp.pth"))

X_train,X_val,X_test,y_train,y_val,y_test = get_data()

X_train=torch.tensor(X_train,dtype=torch.float32)
X_val=torch.tensor(X_val,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)
y_val=torch.tensor(y_val,dtype=torch.float32)  
y_test=torch.tensor(y_test,dtype=torch.float32)  
def val(model,X,y):
    # model.eval()  注释模式？自带的一种切换用于验证的函数
    with torch.no_grad():# 需要我手动关闭梯度计算吗，我都没有写计算梯度的函数，默认计算了

        output=model(X)
        prediction=(output>0).float()  #转换为0，1 ?为什么要float,转换bool类型

        acc=(prediction==y.reshape(-1,1)).float().mean()

    return  acc.item()


val_acc=val(
    model,
    X_val,
    y_val
)


test_acc=val(
    model,
    X_test,
    y_test
)


print("验证集准确率:",val_acc)

print("测试集准确率:",test_acc)


with torch.no_grad():
    output=model(X_test)
    prediction=(output>0).float()
    pred_numpy=prediction.numpy()
    y_numpy=y_test.numpy()
    print(pred_numpy.shape)
    print(y_numpy.shape)
    pred_numpy = prediction.numpy().reshape(-1)
    y_numpy = y_test.numpy().reshape(-1)
    cm=confusion_matrix(y_numpy,pred_numpy)

    print("混淆矩阵：")
    print(cm)

# wrong samples

wrong=np.where(pred_numpy!=y_numpy)[0] #    取第一个数组

print("wrong numburs:")
print(wrong)

for i in wrong :

    print("position :",X_test[i],"TRUE:",y_numpy[i],"prediction:",pred_numpy[i])

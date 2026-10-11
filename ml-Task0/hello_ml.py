

import numpy as np

# hello
print("Hello Machine Learing")
# list  
scores=[89,90,92,97,96,95]

# dictionary  
student_scores={"LIU":95,"ZHANG":90,"HUANG":92,"WANG":89,"LI":93}

# averge  
averge_list=sum(scores)/len(scores)
# 默认遍历的是键值 false:averge_dic=sum(student_scores)/len(student_scores)
averge_dic=sum(student_scores.values())/len(student_scores)
# max  
max_list=max(scores)
max_dic=max(student_scores.values())


print("平均分（列表）：",averge_list)
print("最高分（列表）：",max_list)
print("平均分（字典）：",averge_dic)
print("最高分（字典）：",max_dic)



# 创建矩阵
A=np.array([[1,2,7],[6,5,9]])
B=np.array([[2,4,6],[7,9,8],[7,7,7]])
S=A@B
print("矩阵A：\n",A)
print("矩阵B：\n",B)
print("矩阵A的形状：",A.shape)
print("矩阵B的形状：",B.shape)
print("矩阵A和B的乘积：\n",S)
print("矩阵A和B的乘积矩阵形状：",S.shape)




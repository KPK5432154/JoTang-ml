from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as ps



# 画图  

# 划分
def get_data():
    X,y =make_moons( 
    n_samples=300,
    noise=0.1,
    random_state=42
    )
    X_train,X_temp,y_train,y_temp=train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=4
    )

    X_val,X_test,y_val,y_test=train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=4
    )


    ##print(X.shape)
    ##print(X_train.shape)
    ##print(X_val.shape)
    ##print(X_test.shape)
    return X_train,X_val,X_test,y_train,y_val,y_test

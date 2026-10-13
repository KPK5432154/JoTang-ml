from sklearn.datasets import make_moons
import matplotlib.pyplot as ps


X,y = make_moons(
    n_samples=300,
    noise=0.1,
    random_state=42
)


ps.scatter(
    X[:,0],
    X[:,1],
    c=y
)

ps.show()
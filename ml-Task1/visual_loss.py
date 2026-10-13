import matplotlib.pyplot as ps

import numpy as np
loss=np.load("loss.npy")
ps.plot(loss)

ps.xlabel("time")
ps.ylabel("loss")

ps.show()
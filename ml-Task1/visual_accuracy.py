import matplotlib.pyplot as ps

import numpy as np
acc=np.load("acc.npy")
ps.plot(acc)

ps.xlabel("time")
ps.ylabel("acc")

ps.show()
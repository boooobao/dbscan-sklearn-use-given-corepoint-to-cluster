import numpy as np
import matplotlib.pyplot as plt
import readdata

def plotdata(data):
    ar = np.array(data)
    plt.scatter(ar[:, 1], ar[:, 2], marker='o')
    plt.show()
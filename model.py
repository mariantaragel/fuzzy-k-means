import numpy as np
import pandas as pd

class Model:

    train_data = np.array([[1, 2], [1, 4], [1, 1],
                  [8, 5], [8, 6], [8, 3],
                  [10, 8], [10, 6], [10, 10]])
    k_clusters = 3
    q = 2.0

    def __init__(self):
        self.load_data("s1.txt")
        self.datax = self.train_data.T[0]
        self.datay = self.train_data.T[1]

    def load_data(self, filename):
        df = pd.read_csv(filename, sep="    ", header=None, engine="python")
        self.train_data = df.to_numpy()

import numpy as np

class Model:

    train_data = np.array([[1, 2], [1, 4], [1, 1],
                  [8, 5], [8, 6], [8, 3],
                  [10, 8], [10, 6], [10, 10]])
    k_clusters = None
    q = 2.0

    def __init__(self):
        self.datax = self.train_data.T[0]
        self.datay = self.train_data.T[1]
import numpy as np
import pandas as pd

class Model:

    k_clusters = 5
    q = 2.0

    def __init__(self):
        means = np.array([[2, 1], [3, 15], [9, 10], [16, 1], [20, 22]])
        covs = np.array([[[4, 0], [0, 1]], [[1, 0], [0, 2]], [[1, 0.8], [0.8, 1]], [[1, -0.3], [-0.3, 1]], [[1, 0.3], [0.3, 2]]])

        clusters = []
        for i in range(self.k_clusters):
            cluster = np.random.multivariate_normal(means[i], covs[i], 100)
            clusters.append(cluster)

        self.train_data = np.concatenate(clusters)
        self.datax = self.train_data.T[0]
        self.datay = self.train_data.T[1]

    def load_data(self, filename):
        df = pd.read_csv(filename, sep=r"\s+", header=None, engine="python")
        self.train_data = df.to_numpy()
        self.datax = self.train_data.T[0]
        self.datay = self.train_data.T[1]

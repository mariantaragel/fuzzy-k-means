import numpy as np

SMALL_VALUE = 0.001

class FuzzyKMeans:

    initialized = False
    finished = False
    centroids = None

    def __init__(self, train_data, k_clusters, q=2):
        self.train_data = train_data
        self.k_clusters = k_clusters
        self.q = q
        self.N = len(self.train_data)

    def init(self):
        memberships = []
        clusters = []

        for n in range(self.N):
            degrees = (np.random.uniform(0.0, 1.0, self.k_clusters))
            s = sum(degrees)
            degrees = degrees / s
            
            index = np.argmax(degrees)
            clusters.append(index)
            memberships.append(degrees)
        
        self.memberships = np.array(memberships)
        self.initialized = True

        return np.array(clusters)

    def step(self):
        new_centroids = []
        
        for k in range(self.k_clusters):
            a = np.choose([k for b in range(len(self.memberships))], self.memberships.T)
            aq = a ** self.q
            centroid = aq.T @ self.train_data / np.sum(aq)
            new_centroids.append(centroid)
        
        new_centroids = np.array(new_centroids)
        if self.centroids is not None:
            if (abs(self.centroids - new_centroids) < SMALL_VALUE).all():
                self.finished = True

        self.centroids = new_centroids

        clusters = self.new_memshp()
        return clusters, self.centroids

    def new_memshp(self):
        memberships = []
        clusters = []

        distances = self._distances(self.train_data, self.centroids) ** (2 / (self.q - 1))

        for i in distances[:, None]:
            for j in i:
                if (j == 0).any():
                    j += SMALL_VALUE
                degrees = 1 / np.sum(i.T / j, axis=1)

                index = np.argmax(degrees)
                clusters.append(index)
                memberships.append(degrees)


        self.memberships = np.array(memberships)
        return np.array(clusters)

    def _distances(self, X, C):
        return np.sqrt(np.einsum("ijk->ij", ((X[:, None, :] - C) ** 2)))

if __name__ == "__main__":
    k_clusters = 3
    train_data = np.array([[1, 2], [1, 4], [1, 1],
                  [5, 5], [5, 7], [5, 3],
                  [10, 8], [10, 6], [10, 10]])
    fkm = FuzzyKMeans(train_data=train_data, k_clusters=k_clusters)
    fkm.init()
    while not fkm.finished:
        fkm.step()
    print("Finished!")
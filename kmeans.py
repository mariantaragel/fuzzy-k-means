import numpy as np

SMALL_VALUE = 0.001

class KMeans:
    
    initialized = False
    finished = False
    centroids = None

    def __init__(self, train_data, k_clusters):
        self.train_data = train_data
        self.k_clusters = k_clusters

    def init(self):
        centroids = np.random.permutation(self.train_data)[:self.k_clusters]
        self.centroids = np.array(centroids)
        self.initialized = True
        return self.centroids

    def step(self):
        clusters = []
        
        for d in self.train_data:
            distances = self._distances(d, self.centroids)
            index = np.argmin(distances)
            clusters.append(index)
        
        self.clusters = np.array(clusters)

        self.new_centers()

        return self.clusters, self.centroids

    def new_centers(self):
        new_centroids = []

        for c in range(self.k_clusters):
            tmp = []
            for index in range(len(self.clusters)):
                if c == self.clusters[index]:
                    tmp.append(self.train_data[index])
            tmp = np.array(tmp)

            new_centroid = np.sum(tmp, axis=0) / len(tmp)
            new_centroids.append(new_centroid)
        
        new_centroids = np.array(new_centroids)
        if self.centroids is not None:
            if (abs(self.centroids - new_centroids) < SMALL_VALUE).all():
                self.finished = True

        self.centroids = new_centroids

    def _distance(self, x, c):
        return np.sqrt(np.sum((x - c) ** 2))
    
    def _distances(self, x, C):
        distances = []
        
        for c in C:
            distances.append(self._distance(x, c))
        
        return np.array(distances)

if __name__ == "__main__":
    k_clusters = 3
    train_data = np.array([[1, 2], [1, 4], [1, 1],
                  [5, 5], [5, 7], [5, 3],
                  [10, 8], [10, 6], [10, 10]])
    km = KMeans(train_data=train_data, k_clusters=k_clusters)
    km.init()
    while not km.finished:
        km.step()
    print("Finished!")
import numpy as np

SMALL_VALUE = 0.01

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
            degrees = np.round(degrees / s, 2)
            if sum(degrees) > 1.0:
                degrees[-1] -= SMALL_VALUE
            if sum(degrees) < 1.0:
                degrees[-1] += SMALL_VALUE
            
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
        if np.equal(self.centroids, new_centroids).all():
            self.finished = True

        self.centroids = new_centroids

        clusters = self.new_memshp()
        return clusters, self.centroids

    def new_memshp(self):
        memberships = []
        clusters = []

        for i in range(self.N):
            mshp = []
            for j in range(self.k_clusters):
                res = 0
                for k in range(self.k_clusters):
                    dist_up = self._distance(self.train_data[i], self.centroids[j])
                    dist_down = self._distance(self.train_data[i], self.centroids[k])
                    if dist_down == 0:
                        dist_down += SMALL_VALUE
                    res += ((dist_up / dist_down) ** (2 / (self.q - 1)))
                res = 1 / res
                mshp.append(res)
            
            mshp = np.round(mshp, 2)
            if sum(mshp) > 1.0:
                mshp[-1] -= SMALL_VALUE
            if sum(mshp) < 1.0:
                mshp[-1] += SMALL_VALUE

            index = np.argmax(mshp)
            clusters.append(index)
            memberships.append(mshp)

        self.memberships = np.array(memberships)
        return np.array(clusters)

    def _distance(self, x, c):
        return np.sqrt(np.sum((x - c) ** 2))


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
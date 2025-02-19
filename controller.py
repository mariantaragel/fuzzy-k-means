from kmeans import KMeans
from fuzzykmeans import FuzzyKMeans

class Controller:

    def __init__(self, model, view):
        self.model = model
        self.view = view

    def init(self, filename=None):
        self.model.k_clusters = self.view.get_k_clusters()
        self.model.q = self.view.get_q()
        if filename != None:
            self.model.load_data(filename)
        
        self.fkm = FuzzyKMeans(train_data=self.model.train_data, k_clusters=self.model.k_clusters, q=self.model.q)
        self.km = KMeans(train_data=self.model.train_data, k_clusters=self.model.k_clusters)

        self.view.b2["state"] = "active"
        self.view.draw_points(self.model.datax, self.model.datay)

    def step(self):
        if self.view.radvar.get() == "fuzzy":
            self.fuzzy_step()
        elif self.view.radvar.get() == "k-means":
            self.kmeans_step()

    def fuzzy_step(self):
        if not self.fkm.initialized:
            clusters = self.fkm.init()
            self.view.draw_points_in_clusters(self.model.datax, self.model.datay, clusters)
        else:
            clusters, centroids = self.fkm.step()
            self.view.draw_points_in_clusters(self.model.datax, self.model.datay, clusters, centroids)

        if self.fkm.finished:
            print("FKM: Finished")
            print("FKM: Centroids are \n", self.fkm.centroids)
            self.view.b2["state"] = "disabled"

    def kmeans_step(self):
        if self.km.initialized == False:
            centroids = self.km.init()
            self.view.draw_centroids(centroids.T[0], centroids.T[1], "red")
        else:
            clusters, centroids = self.km.step()
            self.view.draw_points_in_clusters(self.model.datax, self.model.datay, clusters, centroids)

        if self.km.finished:
            print("KM: Finished")
            print("KM: Centroids are \n", self.km.centroids)
            self.view.b2["state"] = "disabled"
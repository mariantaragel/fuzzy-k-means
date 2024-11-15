import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class View(tk.Frame):

    def __init__(self, root, controller):
        super().__init__(root)
        self.root = root
        self.controller = controller
        self.root.title("Fuzzy k-means")
        self.root.geometry("700x600")
        self.create_buttons()
        self.create_graph()

    def create_graph(self):
        self.fig = plt.figure(figsize=(7, 6), layout="tight")
        self.ax = self.fig.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.ax.set(
            ylim=(0, 11),
            xlim=(0, 11),
            xticks=[],
            yticks=[]
        )
        self.ax.spines[:].set_visible(False)
        self.canvas.get_tk_widget().pack()

    def create_buttons(self):
        self.top_frame = tk.Frame(self.root)

        self.l1 = tk.Label(self.top_frame, text="Number of Clusters:")
        self.l1.grid(row=0, column=0)

        vcmd = (self.register(self.callback))
        self.e1 = tk.Entry(self.top_frame, validate="all", validatecommand=(vcmd, "%P"), width=5)
        self.e1.insert(tk.END,"3")
        self.e1.grid(row=0, column=1)

        self.l2 = tk.Label(self.top_frame, text="Fuzzyness:")
        self.l2.grid(row=1, column=0)

        self.e2 = tk.Entry(self.top_frame, width=5)
        self.e2.insert(tk.END,"2.0")
        self.e2.grid(row=1, column=1)
        
        self.b1 = tk.Button(self.top_frame, text="Init", command=self.controller.init)
        self.b1.grid(row=0, column=2)
        
        self.b2 = tk.Button(self.top_frame, text="Fuzzy Step", command=self.controller.step)
        self.b2.grid(row=0, column=3)
        
        self.b4 = tk.Button(self.top_frame, text="Step KMeans", command=self.controller.kmeans)
        self.b4.grid(row=0, column=4)
        
        self.top_frame.pack()

    def draw_points(self, x, y):
        self.ax.clear()
        self.ax.set(
            ylim=(0, 11),
            xlim=(0, 11),
            xticks=[],
            yticks=[]
        )

        self.ax.scatter(x, y, color="black")

        self.canvas.draw()

    def draw_points_in_clusters(self, x, y, clusters, centroids=None):
        self.ax.clear()
        self.ax.set(
            ylim=(0, 11),
            xlim=(0, 11),
            xticks=[],
            yticks=[]
        )

        for c in np.unique(clusters):
            cluster = np.argwhere(clusters == c)
            cluster_x = x[cluster]
            cluster_y = y[cluster]
            self.ax.scatter(cluster_x, cluster_y)
        if centroids is not None:
            self.ax.scatter(centroids.T[0], centroids.T[1], color="black", marker="x")

        self.canvas.draw()

    def draw_centroids(self, x, y, color=None):
        self.ax.set(
            ylim=(0, 11),
            xlim=(0, 11),
            xticks=[],
            yticks=[]
        )

        if color is None:
            color = "balck"
        self.ax.scatter(x, y, color=color, marker="x")
        
        self.canvas.draw()


    def callback(self, P):
        if str.isdigit(P) or P == "":
            return True
        else:
            return False

    def get_k_clusters(self):
        try:
            out = int(self.e1.get())
        except Exception:
            return 3

        return out
    
    def get_q(self):
        try:
            out = float(self.e2.get())
        except Exception:
            return 2.0
        
        return out
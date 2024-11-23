import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class View(tk.Frame):

    def __init__(self, root, controller):
        super().__init__(root)
        self.root = root
        self.controller = controller
        self.root.title("Fuzzy k-means")
        self.root.geometry("600x500")
        self.create_buttons()
        self.create_graph()

    def create_graph(self):
        self.fig = plt.figure()
        self.ax = self.fig.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.clear_graph()
        self.ax.spines[:].set_visible(False)
        self.canvas.get_tk_widget().pack()

    def clear_graph(self):
        self.ax.clear()
        self.ax.set(
            xticks=[],
            yticks=[]
        )

    def create_buttons(self):
        self.top_frame = tk.Frame(self.root)

        self.l1 = tk.Label(self.top_frame, text="Number of Clusters:")
        self.l1.grid(row=0, column=0)

        vcmd = (self.register(self.callback))
        self.e1 = tk.Entry(self.top_frame, validate="all", validatecommand=(vcmd, "%P"), width=5)
        self.e1.insert(tk.END,"5")
        self.e1.grid(row=0, column=1)

        self.l2 = tk.Label(self.top_frame, text="Fuzziness:")
        self.l2.grid(row=1, column=0)

        self.e2 = tk.Entry(self.top_frame, width=5)
        self.e2.insert(tk.END,"2.0")
        self.e2.grid(row=1, column=1)

        self.l2 = tk.Label(self.top_frame, text="Dataset:")
        self.l2.grid(row=2, column=0)
        
        self.b1 = tk.Button(self.top_frame, text="Init", command=self.controller.init)
        self.b1.grid(row=0, column=2)
        
        self.b2 = tk.Button(self.top_frame, text="Step", command=self.controller.step)
        self.b2.grid(row=0, column=3)

        self.radvar = tk.StringVar()
        
        self.r1 = tk.Radiobutton(self.top_frame, text="Fuzzy", variable=self.radvar, value="fuzzy", command=self.controller.init)
        self.r1.grid(row=0, column=4)
        
        self.r2 = tk.Radiobutton(self.top_frame, text="K-means", variable=self.radvar, value="k-means", command=self.controller.init)
        self.r2.grid(row=0, column=5)

        self.b3 = tk.Button(self.top_frame, text="Open", command=self.open)
        self.b3.grid(row=2, column=1)

        self.radvar.set("fuzzy")
        self.top_frame.pack()

    def open(self):
        filename = filedialog.askopenfile()
        self.controller.init(filename)

    def draw_points(self, x, y):
        self.clear_graph()
        self.ax.scatter(x, y, color="black", alpha=0.3)
        self.canvas.draw()

    def draw_points_in_clusters(self, x, y, clusters, centroids=None):
        self.clear_graph()

        for c in np.unique(clusters):
            cluster = np.argwhere(clusters == c)
            cluster_x = x[cluster]
            cluster_y = y[cluster]
            self.ax.scatter(cluster_x, cluster_y, alpha=0.3)
        if centroids is not None:
            self.ax.scatter(centroids.T[0], centroids.T[1], color="black", marker="x")

        self.canvas.draw()

    def draw_centroids(self, x, y, color=None):
        self.ax.set(
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
            if out > 32:
                print("Warning: Maximum number of clusters is 32")
                self.e1.delete(0, tk.END)
                self.e1.insert(tk.END, "5")
        except Exception:
            self.e1.delete(0, tk.END)
            self.e1.insert(tk.END, "5")
            return 5

        return out
    
    def get_q(self):
        try:
            out = float(self.e2.get())
            if out <= 1.0:
                print("Warning: Fuzziness must be in interval (1, +inf)")
                self.e2.delete(0, tk.END)
                self.e2.insert(tk.END, "2.0")
        except Exception:
            self.e2.delete(0, tk.END)
            self.e2.insert(tk.END, "2.0")
            return 2.0
        
        return out

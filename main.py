import tkinter as tk
from model import Model
from view import View
from controller import Controller

if __name__ == "__main__":
    root = tk.Tk()
    model = Model()
    controller = Controller(model, None)
    view = View(root, controller)
    controller.view = view
    view.mainloop()
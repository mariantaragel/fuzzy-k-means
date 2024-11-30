import tkinter as tk
from model import Model
from view import View
from controller import Controller

def on_closing():
    root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    model = Model()
    controller = Controller(model, None)
    view = View(root, controller)
    controller.view = view
    view.mainloop()
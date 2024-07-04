import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np


class PlotEmbedder:
    def __init__(self, parent, title, row, col):

        self.frame = ttk.Frame(parent)
        self.frame.grid(row=row, column=col, sticky=tk.NSEW)

        self.figure, self.ax = plt.subplots()
        self.ax.set_title(title)

        self.canvas = FigureCanvasTkAgg(self.figure, self.frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_plot(self, x, y):
        self.ax.clear()
        self.ax.plot(x, y)
        self.canvas.draw()

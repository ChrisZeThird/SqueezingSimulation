import numpy as np
import tkinter as tk
from tkinter import ttk

from gui.plot_embedder import PlotEmbedder


class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Function Evolution Visualizer")

        # Parameter controls
        self.parameter_frame = ttk.Frame(root)
        self.parameter_frame.grid(row=0, column=0, columnspan=2, sticky=tk.EW)

        self.param_label = ttk.Label(self.parameter_frame, text="Parameter:")
        self.param_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.param_var = tk.DoubleVar(value=1.0)
        self.param_entry = ttk.Entry(self.parameter_frame, textvariable=self.param_var)
        self.param_entry.pack(side=tk.LEFT, padx=5, pady=5)

        self.update_button = ttk.Button(self.parameter_frame, text="Update Plots", command=self.update_plots)
        self.update_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Configure grid layout
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        # Plot frames in a 2x2 grid
        self.plot_frames = []
        positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
        titles = ["Plot 1", "Plot 2", "Plot 3", "Schema"]

        for pos, title in zip(positions, titles):
            plot_embedder = PlotEmbedder(root, title, pos[0], pos[1])
            self.plot_frames.append(plot_embedder)

        self.update_plots()

    def update_plots(self):
        param_value = self.param_var.get()
        x = np.linspace(0, 10, 100)

        for plot_embedder in self.plot_frames:
            if plot_embedder.ax.get_title() == "Schema":
                y = param_value * np.sin(x)  # Example schema logic
            else:
                y = param_value * np.cos(x)  # Example plot logic

            plot_embedder.update_plot(x, y)

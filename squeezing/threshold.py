import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

import utils.plot_parameters

import numpy as np

from cavity.cavity_formulas import Pump_threshold
from utils.settings import settings


def slider_threshold():
    T = np.linspace(start=0, stop=0.2, num=settings.number_points)
    L = np.linspace(start=0, stop=0.1, num=settings.number_points)

    X = np.linspace(start=0, stop=0.9, num=settings.number_points)
    P = np.linspace(start=0, stop=250, num=settings.number_points) * 1e-3

    TT, LL = np.meshgrid(T, L)

    XX, PP = np.meshgrid(X, P)

    # Initial values
    init_T = 0.2
    init_L = 0.0
    init_P = 150e-3  # 150 mW in Watts

    # Calculate the initial values for E_NL
    # E_NL = ((TT + LL) ** 2) * init_P / 4
    E_NL = (XX ** 2) * PP / 4

    # Create the figure and the line that we will manipulate
    fig = plt.figure()

    ax1 = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_axes((0.1, 0.85, 0.8, 0.1))
    ax1.plot_surface(XX, PP, E_NL, cmap='viridis')
    # Plot the 3D surface
    ax1.set_xlabel('Transmission coefficient (T)', fontsize=18)
    ax1.set_ylabel('Pump threshold power', fontsize=18)
    ax1.set_zlabel('Effective Non-linearity (E_NL)', fontsize=18)
    ax1.set_title('3D Surface Plot')

    # Add a slider for P_thr
    # s = Slider(ax=ax2, label='P_thr (mW)', valmin=0, valmax=250, valinit=init_P * 1e3)
    #
    # # Update function to be called anytime the slider's value changes
    # def update(val):
    #     P_thr = s.val * 1e-3  # Convert from W to mW
    #     E_NL = ((TT + LL) ** 2) * P_thr / 4
    #     ax1.cla()
    #     ax1.plot_surface(T, L, E_NL, cmap='viridis')
    #     # ax.set_xlabel('Transmission coefficient (T)')
    #     # ax.set_ylabel('Loss (L)')
    #     # ax.set_zlabel('Effective Non-linearity (E_NL)')
    #     # ax.set_title('3D Surface Plot')
    #
    # # Call update function when slider value changes
    # s.on_changed(update)

    plt.show()

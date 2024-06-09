import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

import utils.plot_parameters

import numpy as np

from cavity.cavity_formulas import Pump_threshold
from utils.settings import settings


def slider_threshold():
    T = np.linspace(start=0, stop=0.2, num=settings.number_points)
    L = np.linspace(start=0, stop=0.05, num=settings.number_points)

    X = np.linspace(start=0., stop=0.12, num=settings.number_points)
    P = np.linspace(start=100, stop=250, num=settings.number_points) * 1e-3

    TT, LL = np.meshgrid(T, L)
    XX, PP = np.meshgrid(X, P)

    # Calculate the initial values for E_NL
    fixed_P = 150e-3
    E_NL_fixedP = ((TT + LL) ** 2) * fixed_P / 4
    E_NL = (XX ** 2) / (PP * 4)

    # Create the figure and the line that we will manipulate
    # E vs (T,L)
    fig1 = plt.figure(figsize=(9, 9))
    ax1 = fig1.add_subplot(111, projection='3d')

    ax1.set_xlabel('\nTransmission coefficient (T)', fontsize=18)
    ax1.set_ylabel('\nIntracavity loss', fontsize=18)
    ax1.set_zlabel('\nEffective Non-linearity $E_{NL}$ (W$^{-1}$)', fontsize=18, labelpad=20)
    ax1.tick_params(axis='z', which='major', pad=15)

    ax1.plot_surface(TT, LL, E_NL_fixedP, cmap='viridis')

    # E vs (T+L, P)
    fig2 = plt.figure(figsize=(9, 9))
    ax2 = fig2.add_subplot(111, projection='3d')

    ax2.set_xlabel('\nTransmission coefficient (T)', fontsize=18)
    ax2.set_ylabel('\nPump threshold power', fontsize=18)
    ax2.set_zlabel('\nEffective Non-linearity $E_{NL}$ (W$^{-1}$)', fontsize=18, labelpad=20)
    ax2.tick_params(axis='z', which='major', pad=15)

    ax2.plot_surface(XX, PP, E_NL, cmap='viridis')

    plt.tight_layout()
    plt.show()

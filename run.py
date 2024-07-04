from tkinter import Tk, mainloop

from cavity.bandwidth import bandwidth
from cavity.waist import waist, Kaertner

from squeezing.plot import squeezing_vs_pump, squeezing_vs_wavelength
from squeezing.threshold import slider_threshold

from utils.settings import settings

from gui.plot_embedder import PlotEmbedder
from gui.main_gui import MainApplication

if __name__ == '__main__':
    # # -- Squeezing -- #
    # if settings.plot_pump_power:
    #     squeezing_vs_pump()
    #
    # if settings.squeezing_wavelength:
    #     squeezing_vs_wavelength()
    #
    # if settings.plot_threshold:
    #     slider_threshold()
    #
    # # -- Optimize bandwidth -- #
    # if settings.plot_bandwidth:
    #     bandwidth()
    #
    # # -- Optimize waist -- #
    # if settings.plot_waist:
    #     waist()
    #
    # # -- Kaertner class notes -- #
    # if settings.plot_kaertner:
    #     Kaertner()

    root = Tk()
    app = MainApplication(root)
    root.mainloop()

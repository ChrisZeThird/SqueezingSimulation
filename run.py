from cavity.bandwidth import bandwidth
from cavity.waist import waist, plot_from_csv, angle_evolution, plot_w1_w2_vs_L

from squeezing.plot import squeezing_vs_pump, squeezing_vs_wavelength
from squeezing.threshold import slider_threshold

from utils.settings import settings

if __name__ == '__main__':
    # -- Squeezing -- #
    if settings.plot_pump_power:
        squeezing_vs_pump(omega=0)

    if settings.squeezing_wavelength:
        squeezing_vs_wavelength(escape_efficiency=0.55)

    if settings.plot_threshold:
        slider_threshold()

    # -- Optimize bandwidth -- #
    if settings.plot_bandwidth:
        bandwidth()

    # -- Optimize waist -- #
    if settings.plot_waist:
        # waist()
        # plot_from_csv()
        # angle_evolution(L=550e-3, dc=124e-3)
        plot_w1_w2_vs_L(wavelength_nm=780.0)

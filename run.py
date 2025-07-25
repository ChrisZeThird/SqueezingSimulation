from cavity.bandwidth import bandwidth
from cavity.waist import waist, angle_evolution, plot_max_waist_vs_all

from squeezing.plot import squeezing_vs_pump, squeezing_vs_wavelength
from squeezing.threshold import slider_threshold

from utils.settings import settings

import numpy as np

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
        plot_max_waist_vs_all()
        # results = angle_evolution(L_values=np.array([600, 700, 800, 900]) * 1e-3, plot=True)
        # custom_dc_range = np.linspace(start=163, stop=224, num=100) * 1e-3
        # angle_evolution(L_values=np.array([settings.fixed_length]), fixed_theta_deg=np.array([3, 5, 10]), plot=True, dc_range=custom_dc_range)

    if settings.plug_value:
        L = settings.fixed_length
        R = settings.R
        l_crystal = settings.crystal_length
        n = settings.crystal_index
        wavelength = settings.wavelength

        sweep_array = np.linspace(start=settings.d_curved_min, stop=settings.d_curved_max,
                                  num=settings.number_points)

        kwargs = {
            'd_curved': sweep_array,
            'L': L,
            'R': settings.R,
            'l_crystal': settings.crystal_length,
            'index_crystal': settings.crystal_index,
            'wavelength': settings.wavelength
        }
        import cavity.cavity_formulas as cf

        z1, z2, w1, w2, valid_indices = cf.Beam_waist(**kwargs)

        # Find max of w1
        max_idx = np.argmax(w1[valid_indices[0]])
        max_w1 = w1[valid_indices[0]][max_idx]
        associated_w2 = w2[valid_indices[0]][max_idx]

        d_curved = sweep_array[valid_indices[0]][max_idx]

        # Squeezing parameters
        T = 0.08
        L = 0.03
        escape_efficiency = T / (T + L)



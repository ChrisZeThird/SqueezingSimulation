import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

from utils.misc import wavelength_to_omega, convert_np_to_string, arrays_to_tuples_list
from noise_spectrum import noise_spectrum_x, noise_spectrum_p
from utils.settings import settings
from utils.plot_parameters import AnyObjectHandler

# -- Matplotlib parameters -- #
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc('legend.title_fontsize', titlesize=MEDIUM_SIZE)

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# -- Arrays definitions -- #
lambda_array = np.linspace(start=0, stop=1000, num=settings.array_points) * 1e-9  # wavelength
omega_array = np.linspace(start=0, stop=3*settings.omega_c, num=settings.array_points)

escape_efficiencies = np.linspace(start=0.8, stop=0.96, num=10)  # escape efficiency range
escape_efficiency = settings.escape_efficiency

epsilon_array = np.linspace(start=0, stop=1, num=settings.array_points, endpoint=False)  # threshold or pump power

colors = plt.cm.viridis(np.linspace(0, 1, len(escape_efficiencies)))  # Using a colormap for colors

# -- Squeezing and anti-squeezing versus pump power -- #
fig_pump_power, ax = plt.subplots(figsize=(16, 9))
for idx, escape_efficiency in enumerate(escape_efficiencies):
    sx = noise_spectrum_x(omega=0.0, omega_c=settings.omega_c, escape_efficiency=escape_efficiency, epsilon=epsilon_array)
    sp = noise_spectrum_p(omega=0.0, omega_c=settings.omega_c, escape_efficiency=escape_efficiency, epsilon=epsilon_array)
    color = colors[idx]
    ax.plot(epsilon_array**2, 10*np.log10(sx), color=colors[idx])
    ax.plot(epsilon_array**2, 10*np.log10(sp), color=colors[idx], linestyle='--')


# Create the legend (for line style)
legend_quadrature = ax.legend(['$s_x$', '$s_p$'], loc='upper left', bbox_to_anchor=(1.02, 1), title='Quadratures')
for line in legend_quadrature.get_lines():
    line.set_color('black')  # Set the legend lines to black
legend_quadrature._legend_box.align = "left"
legend_quadrature._legend_box.sep = 15
ax.add_artist(legend_quadrature)

# Create custom legend handles for escape efficiencies
legend_eta = ax.legend([(color, '--') for color in colors],
                       [f'${eta:.3f}$' for eta in escape_efficiencies],
                       handler_map={tuple: AnyObjectHandler()},
                       loc='lower left',
                       bbox_to_anchor=(1.02, 0),
                       title='Escape efficiency')
legend_eta._legend_box.align = "left"
legend_eta._legend_box.sep = 15
ax.add_artist(legend_eta)

ax.set_xlabel("$P/P_{thr} = \epsilon^2$")
ax.set_ylabel('$S$ (dB)')
ax.set_title(f'Squeezing and anti-squeezing versus pump power at zero frequency.', pad=20)

# Adjust the layout to accommodate both legends
plt.subplots_adjust(right=0.85)
plt.grid(True)
plt.show()

# -- Squeezing and anti-squeezing versus wavelength -- #
squeezing_wavelength = False
if squeezing_wavelength:
    fig2 = plt.figure(figsize=(16, 9))
    epsilon = 0.1

    sx = noise_spectrum_x(omega=omega_array, omega_c=settings.omega_c, escape_efficiency=escape_efficiency, epsilon=epsilon)
    sp = noise_spectrum_p(omega=omega_array, omega_c=settings.omega_c, escape_efficiency=escape_efficiency, epsilon=epsilon)
    plt.plot(omega_array/settings.omega_c, 10*np.log10(sx))
    plt.plot(omega_array/settings.omega_c, 10*np.log10(sp), linestyle='--')

    # Create the legend (for line style)
    legend_quadrature = plt.legend(['$s_x$', '$s_p$'], loc='upper left', bbox_to_anchor=(1.0, 1), title='Quadratures')

    plt.xlabel("$\omega/\omega_c$")
    plt.ylabel('$S$ (dB)')
    # plt.title(f'Squeezing and anti-squeezing versus noise frequency ($\eta =$ {escape_efficiency} and $\epsilon =$ {epsilon})')

    # Adjust the layout to accommodate both legends
    plt.subplots_adjust(right=0.85)
    plt.grid(True)
    plt.show()

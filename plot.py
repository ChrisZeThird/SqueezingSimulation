import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

from utils.misc import wavelength_to_omega
from noise_spectrum import noise_spectrum_x, noise_spectrum_p, negativity
from utils.settings import settings

# -- Arrays definition -- #
lambda_array = np.linspace(start=0, stop=800e-9, num=settings.array_points)  # wavelength
omega_array = wavelength_to_omega(lambda_array)

escape_efficiencies = np.linspace(start=0.8, stop=0.96, num=10)  # escape efficiency range
epsilon_array = np.linspace(start=0, stop=1, num=settings.array_points, endpoint=False)  # threshold or pump power

colors = plt.cm.viridis(np.linspace(0, 1, len(escape_efficiencies)))  # Using a colormap for colors

# -- Squeezing and anti-squeezing versus pump power -- #
fig1 = plt.figure(figsize=(16, 9))
for idx, escape_efficiency in enumerate(escape_efficiencies):
    sx = noise_spectrum_x(omega=0.0, omega_c=settings.omega_c, escape_efficiency=escape_efficiency, epsilon=epsilon_array)
    sp = noise_spectrum_p(omega=0.0, omega_c=settings.omega_c, escape_efficiency=escape_efficiency, epsilon=epsilon_array)
    color = colors[idx]
    plt.plot(epsilon_array**2, 10*np.log10(sx), color=colors[idx])
    plt.plot(epsilon_array**2, 10*np.log10(sp), color=colors[idx], linestyle='--')

# Create the legend (for line style)
legend_quadrature = plt.legend(['$s_x$', '$s_p$'], loc='upper left')
for line in legend_quadrature.get_lines():
    line.set_color('black')  # Set the legend lines to black

plt.xlabel("$P/P_{thr} = \epsilon^2$")
plt.ylabel('$S$ (dB)')
plt.title('Squeezing and anti-squeezing versus pump power at zero frequency')

# Adjust the layout to accommodate both legends
plt.subplots_adjust(right=0.85)
plt.grid(True)
plt.show()

# -- Squeezing and anti-squeezing versus wavelength -- #
fig2 = plt.figure(figsize=(16, 9))
sx = noise_spectrum_x(omega=omega_array, omega_c=settings.omega_c, escape_efficiency=0.8, epsilon=0.5)
sp = noise_spectrum_p(omega=omega_array, omega_c=settings.omega_c, escape_efficiency=0.8, epsilon=0.5)
plt.plot(omega_array/settings.omega_c, 10*np.log10(sx))
plt.plot(omega_array/settings.omega_c, 10*np.log10(sp), linestyle='--')

# Create the legend (for line style)
legend_quadrature = plt.legend(['$s_x$', '$s_p$'], loc='upper right')

plt.xlabel("$\omega/\omega_c$")
plt.ylabel('$S$ (dB)')
plt.title('Squeezing and anti-squeezing versus frequency ($\eta = 0.8$ and $\epsilon = 0.5$)')

# Adjust the layout to accommodate both legends
plt.subplots_adjust(right=0.85)
plt.grid(True)
plt.show()

# -- Wigner negativity vs wavelength -- #
# Omega = np.linspace(start=0, stop=1, num=settings.array_points)
# sx = noise_spectrum_x(omega=0, omega_c=1, escape_efficiency=0.8, epsilon=epsilon_array)
# sp = noise_spectrum_p(omega=0, omega_c=1, escape_efficiency=0.8, epsilon=epsilon_array)
# wigner = negativity(noise_x=sx, noise_p=sp)
#
# fig3 = plt.figure(figsize=(16, 9))
# plt.plot(epsilon_array**2, wigner)
# plt.show()

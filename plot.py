import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

from utils.misc import wavelength_to_omega
from noise_spectrum import noise_spectrum_x, noise_spectrum_p
from utils.settings import settings

# -- Arrays definition -- #
lambda_array = np.linspace(start=0, stop=1000, num=settings.array_points)  # wavelength
omega_array = wavelength_to_omega(lambda_array)

escape_efficiencies = np.linspace(start=0.8, stop=0.91, num=5)  # escape efficiency range
epsilon_array = np.linspace(start=0, stop=1, num=settings.array_points, endpoint=False)  # threshold or pump power

colors = plt.cm.viridis(np.linspace(0, 1, len(escape_efficiencies)))  # Using a colormap for colors

# -- Squeezing and anti-squeezing versus pump power -- #
fig = plt.figure(figsize=(16, 9))
for idx, escape_efficiency in enumerate(escape_efficiencies):
    sx = noise_spectrum_x(omega=0.0, omega_c=settings.omega_c, escape_efficiency=escape_efficiency, epsilon=np.sqrt(epsilon_array))
    sp = noise_spectrum_p(omega=0.0, omega_c=settings.omega_c, escape_efficiency=escape_efficiency, epsilon=np.sqrt(epsilon_array))
    color = colors[idx]
    plt.plot(epsilon_array**2, 10*np.log(sx), color=colors[idx], label=f'x quadrature (eta={escape_efficiency})')
    plt.plot(epsilon_array**2, 10*np.log(sp), color=colors[idx], linestyle='--', label=f'p quadrature (eta={escape_efficiency})')
plt.legend(loc='upper right')
plt.xlabel('Epsilon Squared')
plt.ylabel('Noise Level (dB)')
plt.title('Squeezing and Anti-squeezing vs Pump Power')
plt.grid(True)
plt.show()

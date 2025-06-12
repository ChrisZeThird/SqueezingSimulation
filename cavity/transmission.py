import numpy as np
import matplotlib.pyplot as plt

import utils.plot_parameters


# Define parameters
r = 0.95  # field reflectivity
w_ratio = np.linspace(start=0.55, stop=1.5, num=1000)  # omega / FSR

# Define the complex reflection coefficient
exp_term = np.exp(1j * 2 * np.pi * w_ratio)
F = r * (exp_term - 1) / (1 - r**2 * exp_term)

# Calculate reflection intensity
intensity = np.abs(F)**2

# Calculate phase
phase = np.angle(F)

# Calculate derivative of reflection intensity
d_intensity = np.gradient(intensity, w_ratio)

# Create separate plots
plt.figure(figsize=(8, 4))
plt.plot(w_ratio, intensity, label='Reflection Intensity |F(ω)|²')
plt.xlabel('ω / FSR')
plt.ylabel('Intensity')
plt.grid(True)
plt.title('Reflection Intensity')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(w_ratio, phase, label='Reflection Phase arg(F(ω))', color='orange')
plt.xlabel('ω / FSR')
plt.ylabel('Phase [rad]')
plt.grid(True)
plt.title('Reflection Phase')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(w_ratio, d_intensity, label='d|F(ω)|² / d(ω / FSR)', color='green')
plt.xlabel('ω / FSR')
plt.ylabel('d(Intensity)/d(ω/FSR)')
plt.grid(True)
plt.title('Derivative of Reflection Intensity')
plt.tight_layout()
plt.show()

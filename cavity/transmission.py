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
# Split the phase data into positive and negative parts
positive_phase_indices = phase >= 0
negative_phase_indices = phase < 0

# Calculate derivative of reflection intensity
d_intensity = np.gradient(intensity, w_ratio)

# Create separate plots
plt.figure(figsize=(8, 8))
plt.plot(w_ratio, intensity, label='Reflection Intensity |F(ω)|²')
plt.xlabel('ω / FSR')
plt.ylabel('Intensity')
plt.grid(True)
plt.title('Reflection Intensity')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(w_ratio[positive_phase_indices], np.rad2deg(phase[positive_phase_indices]), color='blue')
plt.plot(w_ratio[negative_phase_indices], np.rad2deg(phase[negative_phase_indices]), color='blue')
plt.xlabel('ω / FSR')
plt.ylabel('Phase [°]')
plt.grid(True)
plt.title('Reflection Phase')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(w_ratio, d_intensity, label='d|F(ω)|² / d(ω / FSR)', color='green')
plt.xlabel('ω / FSR')
plt.ylabel('d(Intensity)/d(ω/FSR)')
plt.grid(True)
plt.title('Derivative of Reflection Intensity')
plt.tight_layout()
plt.show()

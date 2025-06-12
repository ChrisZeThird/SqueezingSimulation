import numpy as np
import matplotlib.pyplot as plt

# Mirror power transmission coefficients (example values)
T1 = 0.05  # 5% transmission
T2 = 0.05
R1 = 1 - T1
R2 = 1 - T2

# Frequency axis (normalized by FSR)
omega_FSR = np.linspace(start=0.5, stop=1.5, num=500)
# Convert normalized frequency to phase (since FSR = 2π/L, we let kL = 2π(omega/FSR))
kL = 2 * np.pi * omega_FSR

# Transmission intensity
T = (T1 * T2) / (1 + R1 * R2 - 2 * np.sqrt(R1 * R2) * np.cos(kL))

# Reflection intensity
R = 1 - T

# Derivative of reflection
dR_dnu = np.gradient(R, omega_FSR)

# Plotting T and R
plt.figure(figsize=(10, 5))
plt.plot(omega_FSR, T, label='Transmission $T$', color='blue')
plt.plot(omega_FSR, R, label='Reflection $R$', color='orange')
plt.xlabel(r'$\omega / \mathrm{FSR}$', fontsize=12)
plt.ylabel('Intensity', fontsize=12)
plt.title('Cavity Transmission and Reflection', fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting derivative of reflection
plt.figure(figsize=(10, 4))
plt.plot(omega_FSR, dR_dnu, label=r"$\frac{dR}{d(\omega/\mathrm{FSR})}$", color='green')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel(r'$\omega / \mathrm{FSR}$', fontsize=12)
plt.ylabel(r'Derivative of $R$', fontsize=12)
plt.title('Derivative of Reflection Intensity', fontsize=14)
plt.grid(True)
# plt.legend()
plt.tight_layout()
plt.show()

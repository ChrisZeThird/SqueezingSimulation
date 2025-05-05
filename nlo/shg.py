import matplotlib.pyplot as plt
import numpy as np

import utils.plot_parameters as pp


def out_shg_sorensen(input_coupler, loss, input_power, nonlinear_efficiency):
    """
    Second-harmonic output power from Sorensen 1998
    :param input_coupler:
    :param loss:
    :param input_power:
    :param nonlinear_efficiency:
    :return:
    """
    a = input_coupler + loss
    rho = 4 * input_coupler * input_power * nonlinear_efficiency

    prefactor = a**2 / (9 * nonlinear_efficiency)

    b = 1 + (27 / 2) * (rho / a**3) * (1 + np.sqrt(1 + (4 / 27) * (a**3 / rho)))

    res = prefactor * ((b ** (1 / 6)) - (b ** (-1 / 6)))**4
    return res


# Parameters
loss = 0.02
nonlinear_efficiency = 0.02  # /W
input_powers = [0.1, 0.2, 0.4, 0.6, 0.8]  # in Watts
input_couplers = np.linspace(0.001, 0.25, 300)

# Plot (a): Output SHG vs input coupler for various input powers
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
max_outputs = []

for P1 in input_powers:
    outputs = [out_shg_sorensen(T1, loss, P1, nonlinear_efficiency) for T1 in input_couplers]
    plt.plot(input_couplers * 100, outputs, label=f'{int(P1 * 1e3)} mW')
    max_outputs.append(outputs)

# Highlight the region between 12% and 13%
plt.axvline(12, color='gray', linestyle='--', linewidth=1)
plt.axvline(13, color='gray', linestyle='--', linewidth=1)
plt.axvspan(12, 13, color='gray', alpha=0.2, label='12–13% Region')

plt.xlabel('Input Coupler Transmission (%)')
plt.ylabel('SHG Output Power (arb. units)')
plt.title('(a) SHG Output vs Input Coupler')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)

# Find the input coupler that gives maximum output at 800 mW
outputs_800mW = max_outputs[-2]
optimal_ic_index = np.argmax(outputs_800mW)
optimal_ic = input_couplers[optimal_ic_index]

# Plot (b): Output SHG vs input power for optimal input coupler
input_powers_cont = np.linspace(0.01, 1.0, 300)
outputs_b = [out_shg_sorensen(optimal_ic, loss, p, nonlinear_efficiency) for p in input_powers_cont]

# Define a ±10% range around the optimal input coupler
delta = 0.2
T1_variations = np.linspace(start=optimal_ic * (1 - delta), stop=optimal_ic * (1 + delta), num=3)

plt.subplot(1, 2, 2)
for T1 in T1_variations:
    outputs_b = [out_shg_sorensen(T1, loss, p, nonlinear_efficiency) for p in input_powers_cont]
    plt.plot(input_powers_cont * 1000, outputs_b, label=f'T1 = {T1*100:.2f}%')

plt.xlabel('Input Power (mW)')
plt.ylabel('SHG Output Power (arb. units)')
plt.title(f'(b) SHG Output vs Input Power\n(±10% T1 around {optimal_ic*100:.2f}%)')
plt.legend(fontsize=10)
plt.grid(True)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---------------------------
# Parameters
# ---------------------------
w0 = 2e-4         # Beam waist at fiber output (meters)
λ = 780e-9        # Wavelength (meters)
f = 0.011           # Focal length of lens (meters)
z_max = 0.3       # Total propagation distance (meters)
num_points = 1000 # Plot resolution

z = np.linspace(0, z_max, num_points)

# ---------------------------
# Gaussian Beam Functions
# ---------------------------
z_R = np.pi * w0**2 / λ

def q_parameter(z, z_R):
    return z + 1j * z_R

def propagate_q(q0, ABCD):
    A, B, C, D = ABCD
    return (A * q0 + B) / (C * q0 + D)

def beam_radius(q):
    return np.sqrt(-λ / (np.pi * np.imag(1 / q)))

def compute_beam_profile(d):
    z_before = z[z <= d]
    z_after = z[z > d]
    z_after_shifted = z_after - d

    q0 = q_parameter(0, z_R)
    q_at_lens = propagate_q(q0, [1, d, 0, 1])
    q_after_lens = propagate_q(q_at_lens, [1, 0, -1/f, 1])

    w_before = [beam_radius(propagate_q(q0, [1, z_i, 0, 1])) for z_i in z_before]
    w_after = [beam_radius(propagate_q(q_after_lens, [1, z_i, 0, 1])) for z_i in z_after_shifted]

    return z_before, w_before, z_after, w_after

# ---------------------------
# Initial Plot
# ---------------------------
initial_d = 0.05
z_before, w_before, z_after, w_after = compute_beam_profile(initial_d)

fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)

line_before, = ax.plot(z_before * 1000, np.array(w_before) * 1000, label='Before Lens')
line_after, = ax.plot(z_after * 1000, np.array(w_after) * 1000, label='After Lens')
lens_line = ax.axvline(initial_d * 1000, color='k', linestyle='--', label='Lens Position')

ax.set_xlabel('z (mm)')
ax.set_ylabel('Beam Radius w(z) (mm)')
ax.set_title('Gaussian Beam Propagation Through a Lens')
ax.legend()
ax.grid(True)

# ---------------------------
# Slider Setup
# ---------------------------
ax_d = plt.axes([0.25, 0.1, 0.5, 0.03])
slider_d = Slider(ax_d, 'Lens Distance (m)', 0.01, 0.25, valinit=initial_d, valstep=0.001)

def update(val):
    d = slider_d.val
    z_before, w_before, z_after, w_after = compute_beam_profile(d)
    line_before.set_data(z_before * 1000, np.array(w_before) * 1000)
    line_after.set_data(z_after * 1000, np.array(w_after) * 1000)
    lens_line.set_xdata([d * 1000])
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

slider_d.on_changed(update)

plt.show()

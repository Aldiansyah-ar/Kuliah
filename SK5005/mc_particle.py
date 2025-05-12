import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Physical space parameters (in cm)
width_cm = 100
height_cm = 100
num_particles = 1000
num_steps = 500
step_size_cm = 1
hole_width_cm = 1
particle_radius_cm = 0.3  # radius of each particle (so 1 cm diameter)

# Random hole on an edge
edge = np.random.choice([0, 1, 2, 3])
if edge == 0:  # Top edge
    x_range = np.random.uniform(0, width_cm - hole_width_cm)
    hole_min = np.array([x_range, height_cm])
    hole_max = np.array([x_range + hole_width_cm, height_cm])
elif edge == 1:  # Bottom edge
    x_range = np.random.uniform(0, width_cm - hole_width_cm)
    hole_min = np.array([x_range, 0])
    hole_max = np.array([x_range + hole_width_cm, 0])
elif edge == 2:  # Left edge
    y_range = np.random.uniform(0, height_cm - hole_width_cm)
    hole_min = np.array([0, y_range])
    hole_max = np.array([0, y_range + hole_width_cm])
else:  # Right edge
    y_range = np.random.uniform(0, height_cm - hole_width_cm)
    hole_min = np.array([width_cm, y_range])
    hole_max = np.array([width_cm, y_range + hole_width_cm])

# Initialize particles at random positions
positions = np.random.uniform([0, 0], [width_cm, height_cm], size=(num_particles, 2))
active_mask = np.ones(num_particles, dtype=bool)
particle_counts = [np.sum(active_mask)]

# Set up figure and subplots
fig, (ax_particles, ax_count) = plt.subplots(1, 2, figsize=(12, 6))

# Particle scatter plot
sc = ax_particles.scatter([], [], s=(particle_radius_cm * 20)**2, alpha=0.5)
ax_particles.set_xlim(0, width_cm)
ax_particles.set_ylim(0, height_cm)
ax_particles.set_aspect('equal')
ax_particles.set_title('Gas Leakage Simulation (Particles)')
ax_particles.set_xlabel('X (cm)')
ax_particles.set_ylabel('Y (cm)')
ax_particles.grid(True)

# Draw the hole as a red bar
if edge in [0, 1]:
    ax_particles.plot(
        [hole_min[0], hole_max[0]],
        [hole_min[1], hole_max[1]],
        'r-', linewidth=3, label='Leak Hole'
    )
else:
    ax_particles.plot(
        [hole_min[0], hole_max[0]],
        [hole_min[1], hole_max[1]],
        'r-', linewidth=3, label='Leak Hole'
    )
ax_particles.legend()

# Particle count line chart
line, = ax_count.plot([], [], 'g-')
ax_count.set_xlim(0, num_steps)
ax_count.set_ylim(0, num_particles)
ax_count.set_title('Remaining Particles Over Time')
ax_count.set_xlabel('Time Step')
ax_count.set_ylabel('Particles Remaining')
ax_count.grid(True)

# Collision detection with hole (considering particle radius)
def is_in_hole(pos):
    x, y = pos[:, 0], pos[:, 1]
    if edge == 0:  # Top
        return (y + particle_radius_cm >= height_cm) & (x >= hole_min[0]) & (x <= hole_max[0])
    elif edge == 1:  # Bottom
        return (y - particle_radius_cm <= 0) & (x >= hole_min[0]) & (x <= hole_max[0])
    elif edge == 2:  # Left
        return (x - particle_radius_cm <= 0) & (y >= hole_min[1]) & (y <= hole_max[1])
    elif edge == 3:  # Right
        return (x + particle_radius_cm >= width_cm) & (y >= hole_min[1]) & (y <= hole_max[1])
    return np.zeros_like(x, dtype=bool)

# Animation update function
def update(frame):
    global positions, active_mask, particle_counts

    # Move active particles randomly
    steps = np.random.choice([-1, 0, 1], size=(num_particles, 2)) * step_size_cm
    positions[active_mask] += steps[active_mask]

    # Keep within grid
    positions = np.clip(positions, [0, 0], [width_cm, height_cm])

    # Absorb particles entering hole
    entered_hole = is_in_hole(positions)
    active_mask &= ~entered_hole

    # Update particle plot
    sc.set_offsets(positions[active_mask])

    # Update particle count
    particle_counts.append(np.sum(active_mask))
    line.set_data(range(len(particle_counts)), particle_counts)
    ax_count.set_xlim(0, max(len(particle_counts), 10))

    return sc, line

# Start animation
ani = FuncAnimation(fig, update, frames=num_steps, interval=50, blit=True)

plt.tight_layout()
plt.show()

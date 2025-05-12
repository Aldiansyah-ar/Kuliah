import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

class GasSimulation:
    def __init__(
        self,
        width=10.0,
        height=10.0,
        n_particles=100,
        leak_position=(5.0, 0.0),
        leak_size=1.0,
        temperature=1.0,
        dt=0.1
    ):
     
        self.width = width
        self.height = height
        self.n_particles = n_particles
        self.leak_x, self.leak_y = leak_position
        self.leak_size = leak_size
        self.temperature = temperature
        self.dt = dt

        # Inisialisasi posisi dan kecepatan
        self.positions = np.random.rand(n_particles, 2) * np.array([width, height])
        self.velocities = np.random.normal(0, temperature, size=(n_particles, 2)) #n_particle jumlah baris dan 2 untuk kolom vs dan vy
        self.active = np.ones(n_particles, dtype=bool)
        self.escaped_count = 0

    def update(self):
        """
        Satu langkah update posisi dan cek batas + kebocoran pada sisi yang dipilih
        """
        active_idx = np.where(self.active)[0]
        pos = self.positions[active_idx].copy()
        vel = self.velocities[active_idx].copy()

        # Gerakkan partikel
        pos += vel * self.dt

        # Deteksi kebocoran berdasarkan sisi boundary
        if self.leak_y == 0:
            # Bottom
            leak_mask = (pos[:,1] < 0) & (np.abs(pos[:,0] - self.leak_x) <= self.leak_size/2)
        elif self.leak_y == self.height:
            # Top
            leak_mask = (pos[:,1] > self.height) & (np.abs(pos[:,0] - self.leak_x) <= self.leak_size/2)
        elif self.leak_x == 0:
            # Left
            leak_mask = (pos[:,0] < 0) & (np.abs(pos[:,1] - self.leak_y) <= self.leak_size/2)
        elif self.leak_x == self.width:
            # Right
            leak_mask = (pos[:,0] > self.width) & (np.abs(pos[:,1] - self.leak_y) <= self.leak_size/2)
        else:
            # Default bottom
            leak_mask = (pos[:,1] < 0) & (np.abs(pos[:,0] - self.leak_x) <= self.leak_size/2)
        escaped_idx = active_idx[leak_mask]

        # Pantulan elastis di semua boundary kecuali pada area kebocoran
        # Bawah
        reflect_bottom = (pos[:,1] < 0) & ~leak_mask
        pos[reflect_bottom,1] = -pos[reflect_bottom,1]
        vel[reflect_bottom,1] *= -1
        # Atas
        reflect_top = (pos[:,1] > self.height) & ~((self.leak_y == self.height) & leak_mask)
        pos[reflect_top,1] = 2*self.height - pos[reflect_top,1]
        vel[reflect_top,1] *= -1
        # Kiri
        reflect_left = (pos[:,0] < 0) & ~((self.leak_x == 0) & leak_mask)
        pos[reflect_left,0] = -pos[reflect_left,0]
        vel[reflect_left,0] *= -1
        # Kanan
        reflect_right = (pos[:,0] > self.width) & ~((self.leak_x == self.width) & leak_mask)
        pos[reflect_right,0] = 2*self.width - pos[reflect_right,0]
        vel[reflect_right,0] *= -1

        # Tulis kembali posisi dan kecepatan
        self.positions[active_idx] = pos
        self.velocities[active_idx] = vel

        # Tandai dan hitung escaped
        if escaped_idx.size > 0:
            self.active[escaped_idx] = False
            self.escaped_count += escaped_idx.size

    def run_simulation(self, n_steps=1000):
        """
        Jalankan simulasi dan kembalikan riwayat posisi dan jumlah escaped.
        """
        history = {
            'positions': np.zeros((n_steps, self.n_particles, 2)),
            'active': np.zeros((n_steps, self.n_particles), dtype=bool),
            'escaped': np.zeros(n_steps, dtype=int),
        }
        for step in range(n_steps):
            self.update()
            history['positions'][step] = self.positions
            history['active'][step] = self.active
            history['escaped'][step] = self.escaped_count
        return history


def animate_simulation(gas_sim, n_steps=300):
    history = gas_sim.run_simulation(n_steps)
    positions = history['positions']
    active = history['active']
    escaped = history['escaped']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, gas_sim.width)
    ax.set_ylim(0, gas_sim.height)
    ax.set_aspect('equal')
    ax.set_title('Simulasi Kebocoran Gas')

    # Gambar kotak
    ax.add_patch(Rectangle((0, 0), gas_sim.width, gas_sim.height, fill=False, ec='black'))
    # Gambar kebocoran sesuai sisi
    if gas_sim.leak_y == 0:
        leak_patch = Rectangle(
            (gas_sim.leak_x - gas_sim.leak_size/2, 0), gas_sim.leak_size, 0.05, color='red'
        )
    elif gas_sim.leak_y == gas_sim.height:
        leak_patch = Rectangle(
            (gas_sim.leak_x - gas_sim.leak_size/2, gas_sim.height-0.05), gas_sim.leak_size, 0.05, color='red'
        )
    elif gas_sim.leak_x == 0:
        leak_patch = Rectangle(
            (0, gas_sim.leak_y - gas_sim.leak_size/2), 0.05, gas_sim.leak_size, color='red'
        )
    elif gas_sim.leak_x == gas_sim.width:
        leak_patch = Rectangle(
            (gas_sim.width-0.05, gas_sim.leak_y - gas_sim.leak_size/2), 0.05, gas_sim.leak_size, color='red'
        )
    ax.add_patch(leak_patch)

    scatter = ax.scatter([], [], s=10, alpha=0.7)
    info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, verticalalignment='top')

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        info_text.set_text(f"Langkah: 0\nAktif: {np.sum(active[0])}\nEscaped: {escaped[0]}")
        return scatter, info_text

    def update_frame(frame):
        pts = positions[frame][active[frame]]
        scatter.set_offsets(pts if pts.size else np.empty((0, 2)))
        info_text.set_text(f"Langkah: {frame}\nAktif: {active[frame].sum()}\nEscaped: {escaped[frame]}")
        return scatter, info_text

    anim = FuncAnimation(fig, update_frame, frames=n_steps, init_func=init, interval=50, blit=False)
    return anim

if __name__ == '__main__':
    sim = GasSimulation(
        width=10, height=10, n_particles=1000,
        leak_position=(10, 5), leak_size=1,
        temperature=0.5, dt=0.5
    )
    ani = animate_simulation(sim, n_steps=10000)

    # Save the animation as a GIF
    writergif = PillowWriter(fps=30)
    ani.save('gas_simulation.gif', writer=writergif)

    plt.show()

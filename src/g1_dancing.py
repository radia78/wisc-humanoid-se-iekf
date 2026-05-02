from whse.sim.simulation import Simulator

if __name__ == "__main__":
    sim = Simulator(
        npz_path="/Users/radiakbar/Documents/Projects/wisc-humanoid-se-iekf/data/dance_motion.npz",
        show_viewer=True,
    )
    sim.run()
